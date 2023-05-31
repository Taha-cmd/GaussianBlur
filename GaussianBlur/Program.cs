using OpenCL.Net;
using System;
using System.Drawing;

namespace GaussianBlur
{
    class Program
    {
        private const string InputFile = "colors.jpg";
        private const string OutputFile = $"output_{InputFile}";
        private const string KernelSourceFilename = "kernel.cl";
        private const string KernelName = "GaussianBlur";
        private const int Sigma = 1;

        static void Main(string[] args)
        {
            var inputImageBitmap = new Bitmap($"InputImages/{InputFile}");

            using (OpenCLContext openClContext = new OpenCLContext())
            {
                int inputBufferSize = inputImageBitmap.Height * inputImageBitmap.Width;
                float4[] inputImageData = ImageHelper.GetImageData(inputImageBitmap);

                IMem<float4> inputBuffer = openClContext.CreateBuffer<float4>(MemFlags.ReadOnly | MemFlags.CopyHostPtr, inputImageData);
                IMem<float4> outputBuffer = openClContext.CreateBuffer<float4>(MemFlags.WriteOnly, inputBufferSize);

                Kernel kernel = openClContext.GetKernelFromSourceFile(KernelSourceFilename, KernelName);

                var globalSize = new[] { new IntPtr(inputImageBitmap.Width), new IntPtr(inputImageBitmap.Height) };

                // Run the column-wise pass
                openClContext.ExecuteKernel(kernel, 2, globalSize, true,
                    KernelArg.Get(inputBuffer), KernelArg.Get(outputBuffer), KernelArg.Get(inputImageBitmap.Width), KernelArg.Get(inputImageBitmap.Height), KernelArg.Get(Sigma), KernelArg.Get(1));

                var íntermediateResult = openClContext.ReadBuffer(outputBuffer, inputBufferSize);
               IMem<float4> intermediateInputBuffer = openClContext.CreateBuffer<float4>(MemFlags.ReadOnly | MemFlags.CopyHostPtr, íntermediateResult);

                // Run the row-wise pass
                openClContext.ExecuteKernel(kernel, 2, globalSize, true,
                    KernelArg.Get(intermediateInputBuffer), KernelArg.Get(outputBuffer), KernelArg.Get(inputImageBitmap.Width), KernelArg.Get(inputImageBitmap.Height), KernelArg.Get(Sigma), KernelArg.Get(2));

                // Read the result from the output buffer
                float4[] outputImageData = openClContext.ReadBuffer(outputBuffer, inputBufferSize);
                Bitmap outputImageBitmap = ImageHelper.CreateImageFromData(outputImageData, inputImageBitmap.Width, inputImageBitmap.Height);

                outputImageBitmap.Save(OutputFile);

                Console.WriteLine($"Done. Output: {OutputFile}");
                Console.WriteLine("Press any key to exit");
                Console.ReadKey();
            }
        }
    }
}
