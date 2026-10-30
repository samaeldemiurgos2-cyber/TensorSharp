using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

internal static class Program
{
    private sealed record Shape(string Name, int Inner, int Columns, string[] Weights);
    private static readonly Shape[] Shapes =
    [
        new("bf16_ple",512,12288,["per_layer_model_proj.weight"]),
        new("q8_output",512,768,["output.weight"]),
        new("q8_local_qkv",512,2048,["blk.0.attn_q.weight","blk.0.attn_k.weight","blk.0.attn_v.weight"]),
        new("q8_global_qkv",512,3072,["blk.5.attn_q.weight","blk.5.attn_k.weight","blk.5.attn_v.weight"]),
        new("q8_local_attention",1024,512,["blk.0.attn_output.weight"]),
        new("q8_global_attention",2048,512,["blk.5.attn_output.weight"]),
        new("q8_gate_up",512,4096,["blk.0.ffn_gate.weight","blk.0.ffn_up.weight"]),
        new("q8_down",2048,512,["blk.0.ffn_down.weight"]),
        new("q8_input_gate",512,512,["blk.0.inp_gate.weight"]),
        new("q8_layer_projection",512,512,["blk.0.proj.weight"])
    ];

    private static int Main(string[] args)
    {
        if (args.Length == 0 || args.Contains("--help"))
        {
            Console.WriteLine("--model PATH --output PATH [--source-dir DIR] [--warmup 3] [--repeats 9] [--calls 4] [--device 0] [--quality-only | --weights-only]");
            Console.WriteLine("Standalone Linux CUDA 13 cuBLAS cached F32 layout comparison; no TensorSharp model routing or PTX changes.");
            return 0;
        }
        var options = new Dictionary<string,string>();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] is "--quality-only" or "--weights-only") { options.Add(args[i],"true"); continue; }
            if (!args[i].StartsWith("--") || i + 1 == args.Length) throw new ArgumentException("Expected --option value.");
            options.Add(args[i], args[++i]);
        }
        string model = Path.GetFullPath(options["--model"]), output = Path.GetFullPath(options["--output"]);
        if (File.Exists(output)) throw new IOException("Refusing to overwrite an existing probe report.");
        string source = Path.GetFullPath(options.GetValueOrDefault("--source-dir", Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!));
        int warmup = Number(options,"--warmup",3), repeats = Number(options,"--repeats",9), calls = Number(options,"--calls",4);
        int device = Number(options,"--device",0, minimum:0);
        var report = new Dictionary<string,object?>
        {
            ["completed"] = false, ["cuda_executed"] = false, ["started_utc"] = DateTime.UtcNow, ["platform"] = RuntimeInformation.OSDescription,
            ["architecture"] = RuntimeInformation.ProcessArchitecture.ToString(), ["runtime"] = RuntimeInformation.FrameworkDescription,
            ["environment"] = new[]{"NVIDIA_TF32_OVERRIDE","CUBLAS_WORKSPACE_CONFIG","CUDA_VISIBLE_DEVICES"}
                .ToDictionary(name=>name,name=>Environment.GetEnvironmentVariable(name)),
            ["model"] = model, ["model_sha256"] = FileHash(model), ["source_sha256"] = Directory.Exists(source)
                ? Directory.EnumerateFiles(source).Where(path => path.EndsWith(".cs") || path.EndsWith(".csproj")).ToDictionary(path=>Path.GetFileName(path)!, FileHash) : null,
            ["binary_sha256"] = FileHash(Assembly.GetExecutingAssembly().Location),
            ["configuration"] = new {warmup,repeats,calls,device,beta=0,alpha=1,compute=77,rows=new[]{13,29,256,512,8192}},
            ["scope"] = "Cached F32 cuBLAS GEMM only, actual decoded GGUF weights/Embedding shapes; no encoder/attention/HTTP performance claim.",
            ["limitations"] = new[]{"CUDA event interval per call includes any stream idle time from host submission; it is not a hardware instruction-only profiler.",
                "Host dispatch measures cuBLAS submissions separately; completed wall includes event synchronization.",
                "Weight decode/transposition/upload and temporary cuMemAlloc/free are outside steady GEMM timing and reported separately.",
                "FAST_TF32(77) rounds products and can lose large near-cancellation residuals; pedantic69 is a separate diagnostic.",
                "No global math-mode override, no F16 activation conversion, no shared backend/native/PTX mutation."}
        };
        var quality = new List<object>(); var measurements = new List<object>();
        report["quality"] = quality; report["measurements"] = measurements;
        try
        {
            if (options.ContainsKey("--weights-only"))
            {
                using var reader = new GgufWeights(model);
                foreach (var shape in Shapes)
                {
                    float[] values=reader.Joined(shape.Weights,shape.Inner,shape.Columns);
                    measurements.Add(new{shape.Name,shape.Inner,shape.Columns,shape.Weights,
                        row_major_f32_sha256=FloatHash(values),transposed_f32_sha256=FloatHash(Transpose(values,shape.Inner,shape.Columns))});
                }
                report["scope"]="Read-only GGUF weight-decoder validation; no CUDA calls, device or speed validation.";
                report["completed"] = true; report["finished_utc"] = DateTime.UtcNow; Save(output,report); return 0;
            }
            using (var gpu = new Gpu(device))
            {
                report["cuda_executed"] = true;
                report["hardware"] = new {gpu.Name,gpu.ComputeCapability,gpu.TotalMemory,gpu.DriverVersion,gpu.BlasVersion,
                    cpu=ReadIfExists("/proc/cpuinfo"),cpu_quota=ReadIfExists("/sys/fs/cgroup/cpu.max"),memory_limit=ReadIfExists("/sys/fs/cgroup/memory.max")};
                ulong before = gpu.FreeMemory(); report["free_memory_before_buffers"] = before;
                RunQuality(gpu,quality);
                if (!options.ContainsKey("--quality-only"))
                {
                    using var reader = new GgufWeights(model);
                    foreach (var shape in Shapes)
                    {
                        var prepare = Stopwatch.StartNew(); float[] rowMajor = reader.Joined(shape.Weights,shape.Inner,shape.Columns);
                        float[] transposed = Transpose(rowMajor,shape.Inner,shape.Columns); prepare.Stop();
                        foreach (int rows in new[]{13,29,256,512,8192})
                        {
                            Console.WriteLine($"{shape.Name}: {rows}×{shape.Inner}×{shape.Columns}, compute77");
                            measurements.Add(Benchmark(gpu,shape,rows,rowMajor,transposed,prepare.Elapsed.TotalMilliseconds,warmup,repeats,calls));
                        }
                    }
                }
                gpu.Synchronize(); report["free_memory_after_buffers"] = gpu.FreeMemory();
                report["loaded_library_sha256"] = LoadedLibraries();
            }
            report["completed"] = true; report["finished_utc"] = DateTime.UtcNow;
            Save(output,report); return 0;
        }
        catch(Exception exception)
        {
            report["error"] = exception.ToString(); report["finished_utc"] = DateTime.UtcNow;
            Save(output,report); Console.Error.WriteLine(exception); return 1;
        }
    }
    private static int Number(Dictionary<string,string> options,string name,int fallback,int minimum=1)
    {
        int value=int.Parse(options.GetValueOrDefault(name,fallback.ToString()));
        if(value<minimum || value>10_000)throw new ArgumentOutOfRangeException(name);return value;
    }
    private static string? ReadIfExists(string path)=>File.Exists(path)?File.ReadAllText(path):null;
    private static string FileHash(string path)
    { using var file=File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant(); }
    private static string FloatHash(float[] values)=>Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan()))).ToLowerInvariant();
    private static void Save(string path,object report)
    { Directory.CreateDirectory(Path.GetDirectoryName(path)!);File.WriteAllText(path,JsonSerializer.Serialize(report,new JsonSerializerOptions{WriteIndented=true})+"\n"); }
    private static Dictionary<string,string> LoadedLibraries()
    {
        var paths=new HashSet<string>();
        foreach(string line in File.ReadLines("/proc/self/maps"))
        {
            int slash=line.IndexOf('/');if(slash<0)continue;string path=line[slash..];
            if(path.Contains("libcuda") || path.Contains("libcublas"))if(File.Exists(path))paths.Add(path);
        }
        return paths.ToDictionary(path=>path,FileHash);
    }
    private static float[] Transpose(float[] rowMajor,int inner,int columns)
    {
        var transposed=new float[rowMajor.Length];
        for(int c=0;c<columns;c++)for(int k=0;k<inner;k++)transposed[k*columns+c]=rowMajor[c*inner+k];
        return transposed;
    }
    private static float[] Inputs(int rows,int inner,float magnitude=1)
    {
        var input=new float[checked(rows*inner)];uint state=0x9e3779b9;
        for(int i=0;i<input.Length;i++)
        {state=unchecked(state*1664525+1013904223);input[i]=magnitude*((int)(state>>8)-8388608)/8388608f;}
        return input;
    }
    private static object Benchmark(Gpu gpu,Shape shape,int rows,float[] weight,float[] transposed,double prepareMs,int warmup,int repeats,int calls)
    {
        float[] input=Inputs(rows,shape.Inner);
        var allocate=Stopwatch.StartNew();
        using var x=new DeviceBuffer(input.Length);using var wN=new DeviceBuffer(weight.Length);
        using var wT=new DeviceBuffer(weight.Length);using var y=new DeviceBuffer(checked(rows*shape.Columns));allocate.Stop();
        var upload=Stopwatch.StartNew();x.Upload(input);wN.Upload(transposed);wT.Upload(weight);upload.Stop();
        Action nn=()=>gpu.Gemm(false,x.Pointer,wN.Pointer,y.Pointer,rows,shape.Inner,shape.Columns,77);
        Action tn=()=>gpu.Gemm(true,x.Pointer,wT.Pointer,y.Pointer,rows,shape.Inner,shape.Columns,77);
        for(int i=0;i<warmup;i++){nn();tn();}gpu.Synchronize();
        var nSamples=new List<object>();var tSamples=new List<object>();var nGpu=new List<double>();var tGpu=new List<double>();
        for(int i=0;i<repeats;i++)
        {
            void Measure(Action call,List<object> samples,List<double> timings)
            {var m=gpu.Measure(call,calls);samples.Add(new{gpu_event_ms_per_call=m.GpuMs,host_dispatch_us_per_call=m.HostUs,completed_wall_ms_per_call=m.CompletedWallMs});timings.Add(m.GpuMs);}
            if(i%2==0){Measure(nn,nSamples,nGpu);Measure(tn,tSamples,tGpu);}
            else{Measure(tn,tSamples,tGpu);Measure(nn,nSamples,nGpu);}
        }
        nn();gpu.Synchronize();float[] nOutput=y.Download();tn();gpu.Synchronize();float[] tOutput=y.Download();
        var nQuality=Compare(input,weight,nOutput,rows,shape.Inner,shape.Columns,false,77);
        var tQuality=Compare(input,weight,tOutput,rows,shape.Inner,shape.Columns,false,77);
        var free=Stopwatch.StartNew();y.Dispose();wT.Dispose();wN.Dispose();x.Dispose();free.Stop();
        return new {shape.Name,rows,shape.Inner,shape.Columns,shape.Weights,compute=77,beta=0,
            source_weight_layout="GGUF decoded [out,in]",cached_nn_layout="[in,out],opN,N,lda=out",cached_tn_layout="[out,in],opT,N,lda=in",
            input_stride=shape.Inner,output_stride=shape.Columns,weight_sha256=FloatHash(weight),input_sha256=FloatHash(input),
            decode_and_transpose_ms=prepareMs,allocate_ms=allocate.Elapsed.TotalMilliseconds,upload_ms=upload.Elapsed.TotalMilliseconds,free_ms=free.Elapsed.TotalMilliseconds,
            nn=new{samples=nSamples,median_gpu_ms=Median(nGpu),output_sha256=FloatHash(nOutput),quality=nQuality},
            tn=new{samples=tSamples,median_gpu_ms=Median(tGpu),output_sha256=FloatHash(tOutput),quality=tQuality},
            tn_over_nn_gpu_ratio=Median(tGpu)/Median(nGpu)};
    }
    private static double Median(List<double> values){var sorted=values.Order().ToArray();return sorted[sorted.Length/2];}
    private static void RunQuality(Gpu gpu,List<object> reports)
    {
        foreach(var specification in new[]{(Name:"ordinary",Rows:13,Inner:64,Columns:37,Magnitude:1f),
            (Name:"unaligned",Rows:29,Inner:97,Columns:13,Magnitude:1f),
            (Name:"large_range",Rows:13,Inner:64,Columns:37,Magnitude:80000f),
            (Name:"near_cancellation",Rows:33,Inner:512,Columns:129,Magnitude:1f)})
        {
            var s=specification;float[] input=Inputs(s.Rows+2,s.Inner,s.Magnitude),weight=Inputs(s.Columns,s.Inner,.125f);
            bool cancellation=s.Name=="near_cancellation";
            if(cancellation)
            {
                for(int r=0;r<s.Rows+2;r++)for(int k=0;k<s.Inner;k++)input[r*s.Inner+k]=k%2==0?131104:-131072;
                for(int c=0;c<s.Columns;c++)for(int k=0;k<s.Inner;k++)weight[c*s.Inner+k]=.5f;
            }
            float[] transposed=Transpose(weight,s.Inner,s.Columns);
            using var x=new DeviceBuffer(input.Length);using var wN=new DeviceBuffer(weight.Length);
            using var wT=new DeviceBuffer(weight.Length);using var y=new DeviceBuffer(checked((s.Rows+2)*s.Columns));
            x.Upload(input);wN.Upload(transposed);wT.Upload(weight);
            foreach(int compute in new[]{77,69})foreach(bool tn in new[]{false,true})
            {
                y.Upload(Enumerable.Repeat(37f,y.Count).ToArray());
                gpu.Gemm(tn,IntPtr.Add(x.Pointer,s.Inner*4),tn?wT.Pointer:wN.Pointer,IntPtr.Add(y.Pointer,s.Columns*4),s.Rows,s.Inner,s.Columns,compute);
                gpu.Synchronize();float[] actual=y.Download();
                for(int c=0;c<s.Columns;c++)if(actual[c]!=37 || actual[(s.Rows+1)*s.Columns+c]!=37)
                    throw new InvalidOperationException("GEMM wrote outside its output view.");
                float[] xView=input.AsSpan(s.Inner,s.Rows*s.Inner).ToArray();
                float[] yView=actual.AsSpan(s.Columns,s.Rows*s.Columns).ToArray();
                object comparison=Compare(xView,weight,yView,s.Rows,s.Inner,s.Columns,true,compute);
                if(cancellation && compute==69 && yView.Any(value=>value!=4096f))
                    throw new InvalidOperationException("Pedantic F32 lost exact binary near-cancellation residual.");
                reports.Add(new{s.Name,s.Rows,s.Inner,s.Columns,compute,layout=tn?"T,N":"N,N",nonzero_storage_offset=true,
                    beta=0,guards_passed=true,expected_cancellation_residual=cancellation?4096f:(float?)null,comparison});
            }
        }
    }
    private static object Compare(float[] input,float[] weight,float[] actual,int rows,int inner,int columns,bool all,int compute)
    {
        int[] rowSamples=all?Enumerable.Range(0,rows).ToArray():new[]{0,rows/4,rows/2,Math.Max(0,rows-2),rows-1}.Distinct().ToArray();
        int[] columnSamples=all?Enumerable.Range(0,columns).ToArray():new[]{0,1,columns/4,columns/2,columns*3/4,columns-1}.Distinct().ToArray();
        double squared=0,expectedSquared=0,maxError=0,maxF32Error=0,maxScaledError=0;int count=0;
        foreach(int r in rowSamples)foreach(int c in columnSamples)
        {
            double expected=0,absolute=0;float f32=0;
            for(int k=0;k<inner;k++)
            {float a=input[r*inner+k],b=weight[c*inner+k];double product=(double)a*b;expected+=product;absolute+=Math.Abs(product);f32=MathF.FusedMultiplyAdd(a,b,f32);}
            float value=actual[r*columns+c];double error=Math.Abs(value-expected),scaled=error/Math.Max(absolute,1e-20);
            if(!float.IsFinite(value) || error>Math.Max(1e-6,absolute*(compute==69?2e-6:1e-3)))
                throw new InvalidOperationException($"Independent double-reference bound failed: compute{compute},({r},{c}),actual={value:R},expected={expected:R},abs-products={absolute:R}.");
            maxError=Math.Max(maxError,error);maxF32Error=Math.Max(maxF32Error,Math.Abs(value-f32));maxScaledError=Math.Max(maxScaledError,scaled);
            squared+=error*error;expectedSquared+=expected*expected;count++;
        }
        return new {sample_count=count,all_outputs_checked=all,maximum_absolute_error_double=maxError,
            relative_l2_double=Math.Sqrt(squared/Math.Max(expectedSquared,1e-30)),maximum_absolute_error_ordered_f32=maxF32Error,
            maximum_error_over_absolute_product_sum=maxScaledError,bound=compute==69?2e-6:1e-3,
            full_precision_contract=compute==69,passed=true};
    }
}
