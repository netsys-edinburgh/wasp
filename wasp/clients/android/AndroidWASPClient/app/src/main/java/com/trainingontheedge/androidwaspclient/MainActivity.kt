package com.trainingontheedge.androidwaspclient

import android.os.Bundle
import android.text.Editable
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import com.koushikdutta.async.AsyncServer
import com.koushikdutta.async.callback.DataCallback
import com.koushikdutta.async.callback.CompletedCallback
import com.koushikdutta.async.ByteBufferList
import com.koushikdutta.async.AsyncSocket
import com.koushikdutta.async.Util
import com.example.edgeclientlitert.runner.CpuRunner
import com.example.edgeclientlitert.runner.GpuRunner
import com.example.edgeclientlitert.runner.RunnerCache
import com.koushikdutta.async.DataEmitter

import morphling.Morphling.UMessage
import morphling.global_api.GlobalApi
import java.nio.ByteBuffer
import java.nio.ByteOrder

const val CACHE_SIZE = 50

class MainActivity : AppCompatActivity() {

    private lateinit var progressBar: TextView
    private lateinit var ipEditText: EditText
    private lateinit var portEditText: EditText
    private lateinit var connectButton: Button
    private lateinit var waitingTextView: TextView

    private lateinit var gpuRunner: RunnerCache<GpuRunner>
    private lateinit var cpuRunner: RunnerCache<CpuRunner>

    private val appScope = CoroutineScope(Dispatchers.Main + Job())

    private val registry = com.google.protobuf.ExtensionRegistryLite.newInstance()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        registry.add(GlobalApi.computeGemmRequest)

        // init UI elements
        progressBar = findViewById(R.id.progressBar)
        ipEditText = findViewById(R.id.ipEditText)
        portEditText = findViewById(R.id.portEditText)
        connectButton = findViewById(R.id.connectButton)
        waitingTextView = findViewById(R.id.waitingTextView)

        // init ip and port
        ipEditText.text = Editable.Factory.getInstance().newEditable("10.126.50.116")
        portEditText.text = Editable.Factory.getInstance().newEditable("9999")

        showLoadingUI()

        // load runners
        appScope.launch {
            withContext(Dispatchers.Default) {
                gpuRunner = RunnerCache(
                    context = applicationContext,
                    maxSize = CACHE_SIZE,
                    prepopulate = true,
                    runnerFactory = { key -> GpuRunner() }
                )

                cpuRunner = RunnerCache(
                    context = applicationContext,
                    maxSize = CACHE_SIZE,
                    prepopulate = true,
                    runnerFactory = { key -> CpuRunner() }
                )
            }
            showConnectionUI()
        }

        connectButton.setOnClickListener {
            val ip = ipEditText.text.toString().trim()
            val port = portEditText.text.toString().toIntOrNull()
            if (ip.isNotEmpty() && port != null) {
                showStaticUI()
                startTcpClient(ip, port)
            } else {
                Toast.makeText(this, "Please enter a valid IP and port", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cpuRunner.clear()
        gpuRunner.clear()
        appScope.cancel()
    }

    // UI state: loading the runners.
    private fun showLoadingUI() {
        progressBar.visibility = View.VISIBLE
        ipEditText.visibility = View.GONE
        portEditText.visibility = View.GONE
        connectButton.visibility = View.GONE
        waitingTextView.visibility = View.GONE
    }

    // UI state: connection inputs after loading is complete.
    private fun showConnectionUI() {
        progressBar.visibility = View.GONE
        ipEditText.visibility = View.VISIBLE
        portEditText.visibility = View.VISIBLE
        connectButton.visibility = View.VISIBLE
        waitingTextView.visibility = View.GONE
    }

    // UI state: static UI waiting for TCP data.
    private fun showStaticUI() {
        ipEditText.visibility = View.GONE
        portEditText.visibility = View.GONE
        connectButton.visibility = View.GONE
        waitingTextView.visibility = View.VISIBLE
        waitingTextView.text = "Waiting for TCP data..."
    }

    private fun startTcpClient(ip: String, port: Int) {
        appScope.launch(Dispatchers.IO) {
            try {
                AsyncServer.getDefault().connectSocket(ip, port, object : com.koushikdutta.async.callback.ConnectCallback {
                    override fun onConnectCompleted(ex: Exception?, socket: AsyncSocket?) {
                        if (ex != null) {
                            Log.e("TCP", "Connection failed: ${ex.message}")
                            return
                        }
                        Log.d("TCP", "Connected to server!")

                        val incomingBuffer = ByteBufferList()
                        var loadReadTime = System.nanoTime() // timing for loading data
                        var readStartTime = System.nanoTime() // timing for reading data
                        var formattingStartTime = System.nanoTime() // timing for formatting data
                        var writeStartTime = System.nanoTime() // timing for writing data
                        var storeStartTime = System.nanoTime() // timing for storing data

                        socket?.setDataCallback(object : DataCallback {
                            override fun onDataAvailable(emitter: DataEmitter, bb: ByteBufferList) {
                                if (incomingBuffer.isEmpty) {
                                    readStartTime = System.nanoTime() // reset timing for new data
                                }
                                incomingBuffer.add(bb)

                                if (incomingBuffer.remaining() < 8) {
                                    return  // wait for more data
                                }
                                // peek header without consuming
                                val headerBytes = incomingBuffer.peekBytes(8)
                                val header = ByteBuffer.wrap(headerBytes).order(ByteOrder.BIG_ENDIAN)
                                val protoSize = header.getInt()
                                val tensorSize = header.getInt()

                                // check if the full message is available (add this back tensorSize)
                                if (incomingBuffer.remaining() < 8 + protoSize) {
                                    return  // wait for more data
                                }
                                Log.d("TCP", "Received message with proto size $protoSize and tensor size $tensorSize")
                                incomingBuffer.skip(8)

                                // read payloads
                                val protoBytes = ByteArray(protoSize)
                                incomingBuffer.get(protoBytes, 0, protoSize)

                                val startProtoTime = System.nanoTime()
                                val uMessage = UMessage.parseFrom(protoBytes, registry)
                                Log.d("Evaluation", "Parsed proto message in ${System.nanoTime() - startProtoTime} ns")

                                val tensorBytes = ByteArray(tensorSize)

                                // always true for evaluation
                                if (false) {
                                    incomingBuffer.get(tensorBytes, 0, tensorSize)
                                    Log.d(
                                        "Evaluation",
                                        "Read data in ${System.nanoTime() - readStartTime} ns"
                                    )
                                } else {
                                    Log.d("Evaluation", "Read data in ${System.nanoTime() - readStartTime} ns")

                                    loadReadTime = System.nanoTime()
                                    val m = uMessage.body.getExtension(GlobalApi.computeGemmRequest).row.toInt()
                                    val n = uMessage.body.getExtension(GlobalApi.computeGemmRequest).col.toInt()
                                    val k = uMessage.body.getExtension(GlobalApi.computeGemmRequest).hDim.toInt()
                                    val tensorA = loadTensor(m, k)
                                    val tensorB = loadTensor(k, n)
                                    if (tensorA != null) {
                                        System.arraycopy(tensorA, 0, tensorBytes, 0, tensorA.size)
                                    } else {
                                        Log.e("TensorStorage", "Tensor A not found")
                                    }
                                    if (tensorB != null && tensorA != null) {
                                        System.arraycopy(tensorB, 0, tensorBytes, tensorA.size, tensorB.size)
                                    } else {
                                        Log.e("TensorStorage", "Tensor B not found")
                                    }
                                    Log.d("Evaluation", "Loaded data in ${System.nanoTime() - loadReadTime} ns")
                                }


                                val result: ByteArray = computeMatrix(uMessage, tensorBytes)

                                formattingStartTime = System.nanoTime()
                                val resultSize = result.size
                                val resultSizeBuffer = ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN)
                                resultSizeBuffer.putInt(resultSize)
                                resultSizeBuffer.flip()

                                val resultBuffer = ByteBuffer.wrap(result).order(ByteOrder.BIG_ENDIAN)
                                val resultByteBufferList = ByteBufferList()
                                resultByteBufferList.add(resultSizeBuffer)
                                resultByteBufferList.add(resultBuffer)
                                Log.d("Evaluation", "Formatted result in ${System.nanoTime() - formattingStartTime} ns")

                                writeStartTime = System.nanoTime()
                                socket?.write(resultByteBufferList)
                                Log.d("Evaluation", "Wrote data in ${System.nanoTime() - writeStartTime} ns")

                                // simulate storing the tensor inputs for evaluation (will be same size as input since square gemm)
                                storeStartTime = System.nanoTime()
                                storeTensor(result, uMessage.body.getExtension(GlobalApi.computeGemmRequest).row.toInt(), uMessage.body.getExtension(GlobalApi.computeGemmRequest).hDim.toInt())
                                storeTensor(result, uMessage.body.getExtension(GlobalApi.computeGemmRequest).hDim.toInt(), uMessage.body.getExtension(GlobalApi.computeGemmRequest).col.toInt())
                                Log.d("Evaluation", "Stored tensor in ${System.nanoTime() - storeStartTime} ns")
                            }
                        })

                        socket?.setEndCallback(CompletedCallback { ex ->
                            Log.d("TCP", "Socket closed")
                            appScope.launch(Dispatchers.Main) {

                            }
                        })
                    }
                })
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Connection error: ${e.message}", Toast.LENGTH_LONG).show()
                }
                Log.e("TCP error", "Connection error: ${e.message}")
            }
        }
    }

    private fun printProtoMessage(protoBytes: ByteArray) {
        try {
            val uMessage = UMessage.parseFrom(protoBytes)
            Log.d("ProtoMessage", "Received message: $uMessage")
        } catch (e: Exception) {
            Log.e("ProtoMessage", "Error parsing proto message: ${e.message}")
        }
    }

    private fun computeMatrix(uMessage: UMessage, tensorBytes: ByteArray): ByteArray {
        try {
            val gemmReq = uMessage.body.getExtension(GlobalApi.computeGemmRequest)
            val m = gemmReq.row.toInt()
            val n = gemmReq.col.toInt()
            val k = gemmReq.hDim.toInt()

            Log.d("ComputeMatrix", "Matrix dimensions: A is $m x $k, B is $k x $n")

            // get byte buffers from payload
            val floatSize = 4
            val aSize = m * k
            val bSize = k * n
            val aBuffer = ByteBuffer.allocate(aSize * floatSize).order(ByteOrder.LITTLE_ENDIAN)
            aBuffer.put(tensorBytes, 0, aSize * floatSize)
            val bBuffer = ByteBuffer.allocate(bSize * floatSize).order(ByteOrder.LITTLE_ENDIAN)
            bBuffer.put(tensorBytes, aSize * floatSize, bSize * floatSize)

            val startTime = System.nanoTime()
            val result = cpuRunner.get(aBuffer, bBuffer, m, k, n)
            Log.d("ComputeMatrix", "Computed matrix in ${System.nanoTime() - startTime} ns")

            // convert back to byte array from float array
            val resultBuffer = ByteBuffer.allocate(result.size * floatSize).order(ByteOrder.LITTLE_ENDIAN)
            for (value in result) {
                resultBuffer.putFloat(value)
            }
            return resultBuffer.array()
        } catch (e: Exception) {
            Log.e("ComputeMatrix", "Error during matrix computation: ${e.message}")
            return ByteArray(0)
        }
    }

    /**
     * Storing tensor to disk - make the file name / key the matrix size
     */
    private fun storeTensor(tensorBytes: ByteArray, row: Int, col: Int) {
        try {
            val tensorSize = tensorBytes.size
            val fileName = "tensor_${row}_${col}.bin"
            val file = applicationContext.getFileStreamPath(fileName)
            applicationContext.openFileOutput(fileName, MODE_PRIVATE).use { output ->
                output.write(tensorBytes)
            }
            Log.d("TensorStorage", "Stored tensor of size $tensorSize to $fileName")
        } catch (e: Exception) {
            Log.e("TensorStorage", "Error storing tensor: ${e.message}")
        }
    }

    private fun loadTensor(row: Int, col: Int): ByteArray? {
        return try {
            applicationContext.openFileInput("tensor_${row}_${col}.bin").use { input ->
                val byteArray = ByteArray(input.available())
                input.read(byteArray)
                byteArray
            }
        } catch (e: Exception) {
            Log.e("TensorStorage", "Error loading tensor: ${e.message}")
            null
        }
    }
}
