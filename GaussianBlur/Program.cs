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
        private const string KernelName = "gaussian_blur";

        static void Main(string[] args)
        {
            var inputImageBitmap = new Bitmap($"InputImages/{InputFile}");

            using OpenCLContext openClContext = new();

            int inputBufferSize = inputImageBitmap.Height * inputImageBitmap.Width;
            float4[] inputImageData = ImageHelper.GetImageData(inputImageBitmap);

            IMem<float4> inputBuffer = openClContext.CreateBuffer<float4>(MemFlags.ReadOnly | MemFlags.CopyHostPtr, inputImageData);
            IMem<float4> outputBuffer = openClContext.CreateBuffer<float4>(MemFlags.WriteOnly, inputBufferSize);

            Kernel kernel = openClContext.GetKernelFromSourceFile(KernelSourceFilename, KernelName);

            var globalSize = new[] { new IntPtr(inputImageBitmap.Width), new IntPtr(inputImageBitmap.Height) };

            openClContext.ExecuteKernel(kernel, 2, globalSize, true, 
                KernelArg.Get(inputBuffer), KernelArg.Get(outputBuffer), KernelArg.Get(inputImageBitmap.Width), KernelArg.Get(inputImageBitmap.Height));

            float4[] outputImageData = openClContext.ReadBuffer(outputBuffer, inputBufferSize);
            Bitmap outputImageBitmap = ImageHelper.CreateImageFromData(outputImageData, inputImageBitmap.Width, inputImageBitmap.Height);

            outputImageBitmap.Save(OutputFile);

            Console.WriteLine($"Done. Output: {OutputFile}");
            Console.WriteLine("Press any key to exit");
            Console.ReadKey();
        }
    }
}
