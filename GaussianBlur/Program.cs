using System;
using System.Diagnostics;
using System.IO;
using OpenCL.Net;
using System.Drawing;
using ImageFormat = System.Drawing.Imaging.ImageFormat;

namespace GaussianBlur
{
    class Program
    {
        static void CheckStatus(ErrorCode err)
        {
            if (err != ErrorCode.Success)
            {
                Console.WriteLine("OpenCL Error: " + err.ToString());
                throw new Cl.Exception(err);
            }
        }

        static void PrintDeviceInfo(Device device)
        {
            var enums = Enum.GetValues(typeof(DeviceInfo));

            foreach (var infoType in enums)
            {
                var info = Cl.GetDeviceInfo(device, (DeviceInfo)infoType, out var _);
                Console.WriteLine($"{infoType.ToString()}: {info.CastTo<int>()}");
            }

        }

        static float[] GetImageData(Bitmap image)
        {
            // https://stackoverflow.com/questions/18766004/how-to-convert-from-system-drawing-bitmap-to-grayscale-then-to-array-of-doubles
            int width = image.Width;
            int height = image.Height;

            float[] imageData = new float[width * height];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Color pixel = image.GetPixel(x, y);
                    int averageColor = ((pixel.R + pixel.G + pixel.B) / 3);
                    image.SetPixel(x, y, Color.FromArgb(averageColor, averageColor, averageColor));

                    imageData[y * width + x] = image.GetPixel(x,y).R / 255.0f;
                }
            }

            return imageData;
        }

        private static Bitmap CreateImageFromData(float[] imageData, int width, int height)
        {
            Bitmap image = new Bitmap(width, height);
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    float pixelValue = imageData[y * width + x];
                    int grayscale = (int)(pixelValue * 255);
                    grayscale = Math.Max(0, Math.Min(255, grayscale));
                    image.SetPixel(x, y, Color.FromArgb(grayscale, grayscale, grayscale));
                }
            }

            return image;
        }

        private static Bitmap ConvertToColor(float[] outputData, int width, int height, Bitmap inputImage)
        { 
            Bitmap colorImage = new Bitmap(width, height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int index = y * width + x;
                    float grayscale = outputData[index];
                    int originalGrayscale = (int)(grayscale * 255);

                    // Retrieve the original color from the input image
                    Color originalColor = inputImage.GetPixel(x, y);

                    int red = Math.Min((int)(originalColor.R * (originalGrayscale / 255.0f)), 255);
                    int green = Math.Min((int)(originalColor.G * (originalGrayscale / 255.0f)), 255);
                    int blue = Math.Min((int)(originalColor.B * (originalGrayscale / 255.0f)), 255);


                    Color colorPixel = Color.FromArgb(originalColor.A, red, green, blue);


                    // Set the color pixel in the output image
                    colorImage.SetPixel(x, y, colorPixel);
                }
            }

            return colorImage;
        }

        // Helper method to clamp the color values within the valid range
        private static Color ClampColor(Color color)
        {
            int max = 255;
            int red = Math.Max(0, Math.Min(max, color.R));
            int green = Math.Max(0, Math.Min(max, color.G));
            int blue = Math.Max(0, Math.Min(max, color.B));

            return Color.FromArgb(color.A, red, green, blue);
        }

        static void Main(string[] args)
        {

            string inputImageName = "colors.jpg";

            var bitmap = new Bitmap($"InputImages/{inputImageName}");


            // used for checking error status of api calls
            ErrorCode status;

            // retrieve the number of platforms
            uint numPlatforms = 0;
            CheckStatus(Cl.GetPlatformIDs(0, null, out numPlatforms));

            if (numPlatforms == 0)
            {
                Console.WriteLine("Error: No OpenCL platform available!");
                System.Environment.Exit(1);
            }

            // select the platform
            Platform[] platforms = new Platform[numPlatforms];
            CheckStatus(Cl.GetPlatformIDs(1, platforms, out numPlatforms));
            Platform platform = platforms[0];

            // retrieve the number of devices
            uint numDevices = 0;
            CheckStatus(Cl.GetDeviceIDs(platform, DeviceType.All, 0, null, out numDevices));

            if (numDevices == 0)
            {
                Console.WriteLine("Error: No OpenCL device available for platform!");
                System.Environment.Exit(1);
            }

            // select the device
            Device[] devices = new Device[numDevices];
            CheckStatus(Cl.GetDeviceIDs(platform, DeviceType.All, numDevices, devices, out numDevices));
            Device device = devices[0];

            PrintDeviceInfo(device);

            // create context
            Context context = Cl.CreateContext(null, 1, new Device[] { device }, null, IntPtr.Zero, out status);
            CheckStatus(status);

            // create command queue
            CommandQueue commandQueue = Cl.CreateCommandQueue(context, device, 0, out status);
            CheckStatus(status);

            int bufferSize = bitmap.Height * bitmap.Width;// * sizeof(float);
            float[] input = GetImageData(bitmap);
            float[] output = new float[bufferSize];


            IMem<float> inputBuffer = Cl.CreateBuffer<float>(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr, input, out status);
            CheckStatus(status);
            IMem<float> outputBuffer = Cl.CreateBuffer<float>(context, MemFlags.WriteOnly, output.Length, out status); ;
            CheckStatus(status);

            // write data from the input vectors to the buffers
            //CheckStatus(Cl.EnqueueWriteBuffer(commandQueue, inputBuffer, Bool.True, IntPtr.Zero, new IntPtr(bufferSize), input, 0, null, out var _));

            // create the program
            string programSource = File.ReadAllText("kernel.cl");
            OpenCL.Net.Program program = Cl.CreateProgramWithSource(context, 1, new string[] { programSource }, null, out status);
            CheckStatus(status);

            // build the program
            status = Cl.BuildProgram(program, 1, new Device[] { device }, "", null, IntPtr.Zero);
            if (status != ErrorCode.Success)
            {
                InfoBuffer infoBuffer = Cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.Log, out status);
                CheckStatus(status);
                Console.WriteLine("Build Error: " + infoBuffer.ToString());
                System.Environment.Exit(1);
            }

            // create the vector addition kernel
            OpenCL.Net.Kernel kernel = Cl.CreateKernel(program, "gaussian_blur", out status);
            CheckStatus(status);

            // set the kernel arguments

            CheckStatus(Cl.SetKernelArg<float>(kernel, 0, inputBuffer));
            CheckStatus(Cl.SetKernelArg<float>(kernel, 1, outputBuffer));
            CheckStatus(Cl.SetKernelArg(kernel, 2, bitmap.Width));
            CheckStatus(Cl.SetKernelArg(kernel, 3, bitmap.Height));


            // output device capabilities
            IntPtr paramSize;
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, IntPtr.Zero, InfoBuffer.Empty, out paramSize));
            InfoBuffer maxWorkGroupSizeBuffer = new InfoBuffer(paramSize);
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, paramSize, maxWorkGroupSizeBuffer, out paramSize));
            int maxWorkGroupSize = maxWorkGroupSizeBuffer.CastTo<int>();
            Console.WriteLine("Device Capabilities: Max work items in single group: " + maxWorkGroupSize);

            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemDimensions, IntPtr.Zero, InfoBuffer.Empty, out paramSize));
            InfoBuffer dimensionInfoBuffer = new InfoBuffer(paramSize);
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemDimensions, paramSize, dimensionInfoBuffer, out paramSize));
            int maxWorkItemDimensions = dimensionInfoBuffer.CastTo<int>();
            Console.WriteLine("Device Capabilities: Max work item dimensions: " + maxWorkItemDimensions);

            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemSizes, IntPtr.Zero, InfoBuffer.Empty, out paramSize));
            InfoBuffer maxWorkItemSizesInfoBuffer = new InfoBuffer(paramSize);
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemSizes, paramSize, maxWorkItemSizesInfoBuffer, out paramSize));
            IntPtr[] maxWorkItemSizes = maxWorkItemSizesInfoBuffer.CastToArray<IntPtr>(maxWorkItemDimensions);
            Console.Write("Device Capabilities: Max work items in group per dimension:");
            for (int i = 0; i < maxWorkItemDimensions; ++i)
                Console.Write(" " + i + ":" + maxWorkItemSizes[i]);
            Console.WriteLine();

            // execute the kernel
            // ndrange capabilities only need to be checked when we specify a local work group size manually
            // in our case we provide NULL as local work group size, which means groups get formed automatically
            var globalSize = new[] { new IntPtr(bitmap.Width), new IntPtr(bitmap.Height) };
            IntPtr[] localWorkSize = null; //new[] { new IntPtr(64), new IntPtr(64) };
            CheckStatus(Cl.EnqueueNDRangeKernel(commandQueue, kernel, 2, null, globalSize, localWorkSize, 0, null, out var _));
            //Cl.Finish(commandQueue);

            //Cl.EnqueueReadBuffer(commandQueue, outputMem, Bool.True, IntPtr.Zero, imageSize * sizeof(float), outputBuffer, 0, null, out eventObject);

            // read the device output buffer to the host output array
            //var 

            CheckStatus(Cl.EnqueueReadBuffer(commandQueue, outputBuffer, Bool.True, IntPtr.Zero, new IntPtr(bufferSize * sizeof(float)), output, 0, null, out var _));
            Cl.Finish(commandQueue);

            Bitmap outputImage1 = CreateImageFromData(output, bitmap.Width, bitmap.Height);
            Bitmap outputImage2 = ConvertToColor(output, bitmap.Width, bitmap.Height, bitmap);

            outputImage1.Save("outfile1.jpg");
            outputImage2.Save("outfile2.jpg");

            Console.WriteLine("Done");
            // release opencl objects
            CheckStatus(Cl.ReleaseKernel(kernel));
            CheckStatus(Cl.ReleaseProgram(program));
            CheckStatus(Cl.ReleaseMemObject(outputBuffer));
            CheckStatus(Cl.ReleaseMemObject(inputBuffer));
            CheckStatus(Cl.ReleaseCommandQueue(commandQueue));
            CheckStatus(Cl.ReleaseContext(context));

            Console.ReadKey();
        }
    }
}
