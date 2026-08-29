import re
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from wasp.placement.constants import *
from wasp.placement.arguments import *


args = parse_args()
model_meta = args.model_meta

d_model = model_meta["d_model"]
d_ffn = model_meta["d_ffn"]
n_head = model_meta["n_head"]
n_layer = model_meta["n_layer"]

test_cycles = get_gemm_shapes(args)
multipliers = get_batch_gemm_sizes(args)


def parse_optimization_result(m, num_devices):
    A_list = [0] * num_devices
    B_list = [0] * num_devices
    T_list = [0] * num_devices
    Z_list = [0] * num_devices

    tp_cost = 0

    for i, v in enumerate(m.getVars()):
        if v.x > 0:
            # print(f"{v.varName}: {v.x}")
            line = f"{v.varName}: {v.x}"

            # extract both values from the variable name, e.g., beta[18]: 66.0
            match = re.findall(r"\[(\d+)\]: (\d+)", line)
            if match:
                # print(match)
                i, x = match[0]
                i = int(i)
                x = int(x)

                # print(i, x)
                if "beta" in line:
                    B_list[i] = x
                elif "alpha" in line:
                    A_list[i] = x
                elif "z" in line:
                    Z_list[i] = x
                elif "theta" in line:
                    T_list[i] = x

            # extract both values from the variable name, e.g., beta[18]: 66.0
            match = re.findall(r"tp_cost: (\d+)", line)
            if match:
                tp_cost = float(match[0])

    # print(f"{A_list=}\n{B_list=}\n{Z_list=}\n{T_list=}\n{tp_cost=}")
    print(f"{A_list=}\n{B_list=}\n{Z_list=}\n{tp_cost=}")

    A_list = np.array(A_list)
    B_list = np.array(B_list)
    Z_list = np.array(Z_list)
    # T_list = np.array(T_list)

    return A_list, B_list, Z_list


def solve_SHTP(
    name,
    in_dim,
    h_dim,
    out_dim,
    num_devices,
    dl_bw,
    ul_bw,
    device_flops,
    dl_latency,
    ul_latency,
    cache_input=False,
    cache_weight=False,
):
    n_entry_remaining = in_dim * out_dim

    m = gp.Model("test")
    m.setParam("TimeLimit", 180)
    # m.setParam('OutputFlag', 0)
    m.setParam("MIPGap", 0.05)  # 5% gap, no need to be very precise

    # matrix multiplication B x S x H  with H x D
    beta = m.addMVar(num_devices, vtype=GRB.INTEGER, lb=0, ub=out_dim, name="beta")
    alpha = m.addMVar(num_devices, vtype=GRB.INTEGER, lb=0, ub=in_dim, name="alpha")

    ul_const_list = np.zeros(num_devices)
    dl_const_list = np.zeros(num_devices)
    flops_const_list = np.zeros(num_devices)
    for i in range(num_devices):
        ul_const = BYTE_SIZE / MB / ul_bw[i]
        dl_const = h_dim * BYTE_SIZE / MB / dl_bw[i]
        flop_const = 2 * h_dim / TB / device_flops[i]

        ul_const_list[i] = ul_const
        dl_const_list[i] = dl_const
        flops_const_list[i] = flop_const

    m.addConstr(out_dim <= beta.sum(), name="hidden_size_lb")
    m.addConstr(out_dim * num_devices >= beta.sum(), name="hidden_size_ub")
    m.addConstr(in_dim <= alpha.sum(), name="input_size_lb")
    m.addConstr(in_dim * num_devices >= alpha.sum(), name="input_size_ub")

    z = m.addMVar(shape=num_devices, vtype=GRB.BINARY, name="z")
    # Constraints
    for i in range(num_devices):
        m.addConstr((z[i] == 0) >> (alpha[i] + beta[i] == 0))
        m.addConstr((z[i] == 1) >> (alpha[i] - 1 >= 0))
        m.addConstr((z[i] == 1) >> (beta[i] - 1 >= 0))

        # # Enforce alpha[i] and beta[i] to be zero when z[i] = 0
        # m.addConstr(alpha[i] <= M * z[i], name=f"alpha_zero_{i}")
        # m.addConstr(beta[i] <= M * z[i], name=f"beta_zero_{i}")

        # # Enforce alpha[i] and beta[i] to be non-zero when z[i] = 1
        # m.addConstr(alpha[i] >= z[i], name=f"alpha_non_zero_{i}")
        # m.addConstr(beta[i] >= z[i], name=f"beta_non_zero_{i}")

    tp_cost = m.addVar(vtype=GRB.CONTINUOUS, name="tp_cost")
    comm_max = m.addMVar(num_devices, vtype=GRB.CONTINUOUS, name="comm_max")
    # comm_min[i] = max(alpha[i] * h_dim / dl_bw[i]+ beta[i] * h_dim / dl_bw[i], alpha[i] * beta[i] / ul_bw[i], 2 * alpha[i] * beta[i] / device_flops[i])
    for i in range(num_devices):
        # considers the maximum of the three costs, assume overlapping communication and computation
        if cache_input:
            m.addConstr(
                comm_max[i] >= (beta[i] * dl_const_list[i]),
                name=f"max_dl_{i}",
            )
        else:
            m.addConstr(
                comm_max[i] >= ((alpha[i] + beta[i]) * dl_const_list[i]),
                name=f"max_dl_{i}",
            )
        m.addConstr(
            comm_max[i] >= (alpha[i] * beta[i] * ul_const_list[i]),
            name=f"max_ul_{i}",
        )
        m.addConstr(
            comm_max[i] >= (2 * alpha[i] * beta[i] * flops_const_list[i]),
            name=f"max_comp_{i}",
        )

        m.addConstr(tp_cost >= comm_max[i], name=f"tp_cost_{i}")
        # m.addConstr(
        #     tp_cost
        #     >= alpha[i] * h_dim / dl_bw[i]
        #     + beta[i] * h_dim / dl_bw[i]
        #     + alpha[i] * beta[i] / ul_bw[i],
        #     +2 * alpha[i] * beta[i] / device_flops[i],
        #     # + z[i] * (UL_LATENCY[i] + DL_LATENCY[i]),
        #     name="tp_cost_constraint",
        # )

    product_sum = gp.quicksum(alpha[i] * beta[i] for i in range(num_devices))
    m.addConstr(product_sum == n_entry_remaining, "sum_of_products")

    # m.setObjectiveN(tp_cost, index=0, priority=2, abstol=1e-2)
    # m.setObjectiveN(max_comm-min_comm, index=1, priority=3)
    # m.setObjectiveN(-gp.quicksum(z), index=2, priority=1)
    m.setObjective(tp_cost)

    # m.tune()

    start_time = time.time()
    m.optimize()
    end_time = time.time()
    print(name, "Time taken: ", end_time - start_time, "s")
    # objN = m.getObjective(0)
    # obj_val = objN.getValue()
    # print(name, "Objective value: ", obj_val)
    return m


test_results = {}
cal_results = {}
for cycle, (in_dim, h_dim, out_dim) in test_cycles.items():
    print(f"Cycle {cycle}")

    cache_input = "gate" in cycle or "qkvo" in cycle or "weight" in cycle
    m = solve_SHTP(
        cycle,
        in_dim,
        h_dim,
        out_dim,
        args.num_devices,
        args.dl_bw,
        args.ul_bw,
        args.device_flops,
        args.dl_lat,
        args.ul_lat,
        cache_input=cache_input,
    )

    cost = m.ObjVal
    A_list, B_list, Z_list = parse_optimization_result(m, args.num_devices)

    if cache_input:
        device_latency = np.max(
            [
                B_list * h_dim * args.byte_size / MB / args.dl_bw,
                A_list * B_list * args.byte_size / MB / args.ul_bw,
                2 * A_list * B_list * h_dim / TB / args.device_flops,
            ],
            axis=0,
        )
    else:
        device_latency = np.max(
            [
                A_list * h_dim * args.byte_size / MB / args.dl_bw
                + B_list * h_dim * args.byte_size / MB / args.dl_bw,
                A_list * B_list * args.byte_size / MB / args.ul_bw,
                2 * A_list * B_list * h_dim / TB / args.device_flops,
            ],
            axis=0,
        )
    device_latency = device_latency / 1000  # convert to seconds
    print(f"{device_latency=}")
    dist = np.max(device_latency) - np.mean(device_latency)
    percent_dist = dist / np.mean(device_latency)
    print(f"{dist=}")
    print(f"{percent_dist=}")
    # assert np.isclose(cost / 1000, np.max(device_latency), atol=1e-2), f"{np.max(device_latency)=}, {cost=}"

    test_results[cycle] = m.ObjBound / 1000  # convert to seconds
    cal_results[cycle] = np.max(device_latency)
    print(f"{cycle=}, {m.ObjBound=}, {m.ObjVal=}")
    # test_results[cycle] = np.max(device_latency)

# qkv_parallel = 0
# model_forward_cost = 0
# for cycle, cost in test_results.items():
#     if cycle == "qkvo":
#         model_forward_cost += cost * (4 - qkv_parallel)
#     # elif cycle == "mlp-proj" and model_type == "llama":
#     #     model_forward_cost += cost * (2 - mlp_parallel)
#     else:
#         model_forward_cost += cost
print(f"{test_results=}")
model_layer_cost = 0
model_cal_cost = 0
model_forward_cost = 0
for cycle, cost in test_results.items():
    model_layer_cost += cost * multipliers[cycle]
    model_cal_cost += cal_results[cycle] * multipliers[cycle]
    if not "bk" in cycle:
        model_forward_cost += cost * multipliers[cycle]
    print(f"{cycle=}, {cost=}, {multipliers[cycle]=}")
# print(f"{model_forward_cost=}")
model_forward_cost *= n_layer * 3
print(f"{model_forward_cost=}")
# model_bachward_cost = model_forward_cost * 2
# model_step_cost = model_forward_cost + model_bachward_cost
model_step_cost = model_layer_cost * n_layer
model_cal_cost *= n_layer
print(f"{model_step_cost=}s")
print(f"{model_cal_cost=}s")
