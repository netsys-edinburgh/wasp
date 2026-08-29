import re
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx

from wasp.placement.constants import *
from wasp.placement.arguments import *


args = parse_args()
model_meta = args.model_meta

d_model = model_meta["d_model"]
d_ffn = model_meta["d_ffn"]
n_head = model_meta["n_head"]
n_layer = model_meta["n_layer"]

def solve_SHTP(name, num_devices, dl_bw, ul_bw, device_flops, dl_latency, ul_latency, G):
    topo_sorted = list(nx.topological_sort(G))
    print(topo_sorted)
    #top_sorted is a list of vertex ids topologically sorted
    m = gp.Model("test")
    # m.setParam("Cuts", 3)
    # m.setParam("ConcurrentMIP", 4)
    # m.setParam('TimeLimit', 180)
    m.setParam("MIPGap", 0.05)
    '''
    m.setParam('TimeLimit', 30)
    m.setParam("IntFeasTol", 1e-9)
    m.setParam("MIPGap", 0)
    m.setParam("MIPGapAbs", 0)
    m.setParam("OptimalityTol", 1e-9)
    m.setParam("ScaleFlag", 2)
    m.setParam("NumericFocus", 2)
    m.setParam("PSDTol", 0)
    '''

    #map the node ids to their index in top_sorted
    dg_node_list_idx = {}
    num_tasks = len(topo_sorted)
    for list_iter, i in enumerate(topo_sorted):
        dg_node_list_idx[i] = list_iter

    #in_dim, h_dim and out_dim are lists (one per matrix)
    h_dim = [G.nodes[i]['h_dim'] for i in topo_sorted]
    in_dim = [G.nodes[i]['in_dim'] for i in topo_sorted]
    out_dim = [G.nodes[i]['out_dim'] for i in topo_sorted]
    multipliers = [G.nodes[i]['multiplier'] for i in topo_sorted]
    
    max_in_dim = max(in_dim)
    max_out_dim = max(out_dim)

    #here num_tasks is the number of matrices that need to be multipled.
    #A task is a matrix that needs to me multiploed
    beta = m.addMVar((num_tasks, num_devices), vtype=GRB.INTEGER, lb=0, ub=max_in_dim * max_out_dim,  name="beta")
    alpha = m.addMVar((num_tasks, num_devices), vtype=GRB.INTEGER, lb=0, ub=max_in_dim * max_out_dim,  name="alpha")
    z = m.addMVar((num_tasks, num_devices), vtype=GRB.BINARY, name="z")

    for i in range (num_tasks):
        for j in range(num_devices):
            m.addConstr(alpha[i][j] <= in_dim[i])
            m.addConstr(beta[i][j] <= out_dim[i])
            m.addConstr(alpha[i][j] <= M * z[i][j])
            m.addConstr(beta[i][j] <= M * z[i][j])

            # Enforce alpha[i] and beta[i] to be non-zero when z[i] = 1
            m.addConstr(alpha[i][j] >= z[i][j])
            m.addConstr(beta[i][j] >= z[i][j])


    tp_cost = m.addMVar(num_tasks, vtype=GRB.CONTINUOUS, lb=0, name="tp_cost")

    #temp_pred_tp_cost is just a temp variable to store the time taken for all predecessors
    #to complete their tasks
    temp_pred_tp_cost = m.addMVar(num_tasks, vtype=GRB.CONTINUOUS, lb=0, name="tp_cost")
    comm_max = m.addMVar((num_tasks, num_devices), vtype=GRB.CONTINUOUS, lb=0, name="comm_max")

    for i in range(num_tasks):
        for p in G.predecessors(topo_sorted[i]):
            m.addConstr(temp_pred_tp_cost[i] >= tp_cost[dg_node_list_idx[p]])
        for j in range(num_devices):
        # considers the maximum of the three costs, assume overlapping communication and computation
            m.addConstr(comm_max[i][j] >= (alpha[i][j] * h_dim[i] * BYTE_SIZE / MB / dl_bw[j]  + beta[i] * h_dim[i] * BYTE_SIZE / MB / dl_bw[j]), name=f"max_dl_{i}{j}")
            m.addConstr(comm_max[i][j] >= (alpha[i][j] * beta[i][j] * BYTE_SIZE / MB / ul_bw[j]), name=f"max_ul_{i}{j}")
            m.addConstr(comm_max[i][j] >= (2 * alpha[i][j] * beta[i][j] * h_dim[i] / TB / device_flops[j]), name=f"max_comp_{i}{j}")
            #We can only start computing this matrix multiply task once predecessors complete
            m.addConstr(tp_cost[i] >= temp_pred_tp_cost[i] + comm_max[i][j] * multipliers[i], name = f"max_tp_cost")

    product_sum = [0] * num_tasks
    for i in range(num_tasks):
        product_sum[i] = gp.quicksum(alpha[i][j] * beta[i][j] for j in range(num_devices))
        m.addConstr(product_sum[i] == in_dim[i] * out_dim[i])

    # for i in range(num_tasks):
    #     m.setObjectiveN(tp_cost[i], index=1, priority=num_tasks-i)
    m.setObjective(tp_cost[num_tasks-1], sense=GRB.MINIMIZE)
    
    # m.setParam("TuneTimeLimit", 180)
    # m.tune()

    start_time = time.time()
    m.optimize()
    end_time = time.time()
    print(name, "Time taken: ", end_time - start_time, "s")
    return m


def parse_optimization_result(m, num_task, num_devices):
    A_list = np.zeros((num_task, num_devices))
    B_list = np.zeros((num_task, num_devices))
    Z_list = np.zeros((num_task, num_devices))

    tp_cost = 0

    for i, v in enumerate(m.getVars()):
        if v.x > 0:
            print(f"{v.varName}: {v.x}")
            line = f"{v.varName}: {v.x}"

            # extract both values from the variable name, e.g., beta[18]: 66.0
            match = re.findall(r"\[(\d+),(\d+)\]: (\d+)", line)
            if match:
                # print(match)
                i, j, x = match[0]
                i = int(i)
                j = int(j)
                x = int(x)

                # print(i, x)
                if "beta" in line:
                    B_list[i,j] = x
                elif "alpha" in line:
                    A_list[i,j] = x
                elif "z" in line:
                    Z_list[i,j] = x

                    # extract both values from the variable name, e.g., beta[18]: 66.0
            match = re.findall(r"tp_cost: (\d+)", line)
            if match:
                tp_cost = float(match[0])

    print(f"{A_list=}\n{B_list=}\n{Z_list=}\n{tp_cost=}")

    A_list = np.array(A_list).reshape(num_task, num_devices)
    B_list = np.array(B_list).reshape(num_task, num_devices)
    Z_list = np.array(Z_list).reshape(num_task, num_devices)

    return A_list, B_list, Z_list


test_cycles = get_gemm_shapes(args)
multipliers = get_batch_gemm_sizes(args)

def gen_topology():
    G = nx.DiGraph()

    it = 0
    # for layer in range(n_layer):
    layer = 0
    
    def create_node(name, index):
        in_dim, h_dim, out_dim = test_cycles[name]
        multiplier = multipliers[name]
        G.add_node(index, op=f"layer{layer}.{name}", in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, multiplier=multiplier)
        
    
    
    in_dim, h_dim, out_dim = test_cycles["qkvo"]
    multiplier = multipliers["qkvo"]
    G.add_node(it, op=f"layer{layer}.q", in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, multiplier=multiplier)
    G.add_node(it+1, op=f"layer{layer}.k", in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, multiplier=multiplier)
    G.add_node(it+2, op=f"layer{layer}.v", in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, multiplier=multiplier)
    
    create_node("attn", it+3)
    for i in range(3):
        G.add_edge(it+i, it+3)
    
    create_node("attn_proj", it+4)
    G.add_edge(it+3, it+4)
    
    in_dim, h_dim, out_dim = test_cycles["qkvo"]
    multiplier = multipliers["qkvo"]
    G.add_node(it+5, op=f"layer{layer}.o", in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, multiplier=multiplier)
    G.add_edge(it+4, it+5)
    
    it += 6
    
    if model_meta["model_type"] == "opt":
        create_node("mlp-gate", it)
        G.add_edge(it-1, it)
        
        create_node("mlp-proj", it+1)
        G.add_edge(it, it+1)
        
        create_node("bk_input_mlp-proj", it+2)
        create_node("bk_weight_mlp-proj", it+3)
        G.add_edge(it+1, it+2)
        G.add_edge(it+1, it+3)
        
        create_node("bk_input_mlp-gate", it+4)
        create_node("bk_weight_mlp-gate", it+5)
        G.add_edge(it+2, it+4)
        G.add_edge(it+2, it+5)
        
        it += 6
        
    else:
        create_node("mlp-gate", it)
        create_node("mlp-gate1", it+1)
        G.add_edge(it-1, it)
        G.add_edge(it-1, it+1)
        
        create_node("mlp-proj", it+2)
        G.add_edge(it, it+2)
        G.add_edge(it+1, it+2)
        
        create_node("bk_input_mlp-proj", it+3)
        create_node("bk_weight_mlp-proj", it+4)
        G.add_edge(it+2, it+3)
        G.add_edge(it+2, it+4)
        
        create_node("bk_input_mlp-gate", it+5)
        create_node("bk_weight_mlp-gate", it+6)
        G.add_edge(it+3, it+5)
        G.add_edge(it+3, it+6)
        
        create_node("bk_input_mlp-gate1", it+7)
        create_node("bk_weight_mlp-gate1", it+8)
        G.add_edge(it+3, it+7)
        G.add_edge(it+3, it+8)
        
        it += 9
    
    create_node("bk_input_attn_proj", it)
    create_node("bk_weight_attn_proj", it+1)
    G.add_edge(it-2, it)
    G.add_edge(it-2, it+1)
    
    in_dim, h_dim, out_dim = test_cycles["bk_input_qkvo"]
    multiplier = multipliers["bk_input_qkvo"]
    G.add_node(it+2, op=f"layer{layer}.bk_input_o", in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, multiplier=multiplier)
    in_dim, h_dim, out_dim = test_cycles["bk_weight_qkvo"]
    multiplier = multipliers["bk_weight_qkvo"]
    G.add_node(it+3, op=f"layer{layer}.bk_weight_o", in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, multiplier=multiplier)
    G.add_edge(it, it+2)
    G.add_edge(it, it+3)
    
    
    
    create_node("bk_input_attn", it+4)
    create_node("bk_weight_attn", it+5)
    G.add_edge(it+2, it+4)
    G.add_edge(it+2, it+5)
    
    in_dim, h_dim, out_dim = test_cycles["bk_input_qkvo"]
    multiplier = multipliers["bk_input_qkvo"]
    G.add_node(it+6, op=f"layer{layer}.bk_input_q", in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, multiplier=multiplier)
    G.add_node(it+7, op=f"layer{layer}.bk_input_k", in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, multiplier=multiplier)
    G.add_node(it+8, op=f"layer{layer}.bk_input_v", in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, multiplier=multiplier)
    G.add_edge(it+4, it+6)
    G.add_edge(it+4, it+7)
    G.add_edge(it+4, it+8)
    
    in_dim, h_dim, out_dim = test_cycles["bk_weight_qkvo"]
    multiplier = multipliers["bk_weight_qkvo"]
    G.add_node(it+9, op=f"layer{layer}.bk_weight_q", in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, multiplier=multiplier)
    G.add_node(it+10, op=f"layer{layer}.bk_weight_k", in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, multiplier=multiplier)
    G.add_node(it+11, op=f"layer{layer}.bk_weight_v", in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, multiplier=multiplier)
    G.add_edge(it+4, it+9)
    G.add_edge(it+4, it+10)
    G.add_edge(it+4, it+11)
            
    return G
        
g = gen_topology()

import matplotlib.pyplot as plt
nx.draw(g, pos=nx.shell_layout(g), with_labels=True)
plt.savefig("debug.png")

m = solve_SHTP(
    args.model,
    args.num_devices,
    args.dl_bw,
    args.ul_bw,
    args.device_flops,
    args.dl_lat,
    args.ul_lat,
    g,
)

num_tasks = len(list(nx.topological_sort(g)))
A_list, B_list, Z_list = parse_optimization_result(m, num_tasks, args.num_devices)

print(f"{m.ObjBound=}, {m.ObjVal=}")

cost = m.ObjBound / 1000
print(f"{cost=}s")
cost *= n_layer
print(f"Total time: {cost:.2f}s")