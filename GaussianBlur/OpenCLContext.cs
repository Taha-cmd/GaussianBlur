using OpenCL.Net;
using System;
using System.Collections.Generic;
using System.IO;
using Context = OpenCL.Net.Context;
using Kernel = OpenCL.Net.Kernel;

namespace GaussianBlur
{
    public class OpenCLContext : IDisposable
    {
        public Platform Platform { get; }
        public Device Device { get; }
        public Context Context { get; }
        public CommandQueue CommandQueue { get; }

        public int MaxWorkGroupSize { get; }

        private readonly List<Kernel> _kernelsToRelease = new();
        private readonly List<IMem> _buffersToRelease = new();

        public void ExecuteAndCheckStatus(Func<ErrorCode> func)
        {
            CheckStatus(func());
        }

        public OpenCLContext()
        {
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

            Context context = Cl.CreateContext(null, 1, new Device[] { device }, null, IntPtr.Zero, out status);
            CheckStatus(status);

            // create command queue
            CommandQueue commandQueue = Cl.CreateCommandQueue(context, device, 0, out status);
            CheckStatus(status);

            Context = context;
            Platform = platform;
            Device = device;
            CommandQueue = commandQueue;

            // output device capabilities
            // output device capabilities
            IntPtr paramSize;
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, IntPtr.Zero, InfoBuffer.Empty, out paramSize));
            InfoBuffer maxWorkGroupSizeBuffer = new InfoBuffer(paramSize);
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, paramSize, maxWorkGroupSizeBuffer, out paramSize));
            MaxWorkGroupSize = maxWorkGroupSizeBuffer.CastTo<int>();
            Console.WriteLine("Device Capabilities: Max work items in single group: " + MaxWorkGroupSize);
        }

        static void CheckStatus(ErrorCode err)
        {
            if (err != ErrorCode.Success)
            {
                Console.WriteLine("OpenCL Error: " + err.ToString());
                throw new Cl.Exception(err);
            }
        }

        public IMem<T> CreateBuffer<T>(MemFlags flags, T[] input) where T : struct
        {
            IMem<T> buffer = Cl.CreateBuffer<T>(Context, flags, input, out var status);
            CheckStatus(status);

            _buffersToRelease.Add(buffer);

            return buffer;
        }

        public IMem<T> CreateBuffer<T>(MemFlags flags, int length) where T : struct
        {
            IMem<T> buffer = Cl.CreateBuffer<T>(Context, flags, length, out var status);
            CheckStatus(status);

            _buffersToRelease.Add(buffer);

            return buffer;
        }

        public T[] ReadBuffer<T>(IMem<T> mem, int length) where T : unmanaged
        {
            unsafe
            {
                T[] buffer = new T[length];
                IntPtr lengthInBytes = new IntPtr(length * sizeof(T)); 
                ExecuteAndCheckStatus(() => 
                    Cl.EnqueueReadBuffer(CommandQueue, mem, Bool.True, IntPtr.Zero, lengthInBytes, buffer, 0, null, out var _));

                return buffer;
            }
        }

        public void ExecuteKernel(Kernel kernel, uint dimensions, IntPtr[] globalWorkSize, bool blocking, params KernelArg[] args)
        {
            for (uint index = 0; index < args.Length; index++)
            {
                ExecuteAndCheckStatus(() => Cl.SetKernelArg(kernel, index, args[index].Size, args[index].Value));
            }

            // in our case we provide NULL as local work group size, which means groups get formed automatically
            ExecuteAndCheckStatus(() => 
                Cl.EnqueueNDRangeKernel(CommandQueue, kernel, dimensions, null, globalWorkSize, null, 0, null, out var _));

            if (blocking)
            {
                Cl.Finish(CommandQueue);
            }
        }

        public Kernel GetKernelFromSourceFile(string sourceFile, string kernelName)
        {
            ErrorCode status;

            string programSource = File.ReadAllText(sourceFile);
            OpenCL.Net.Program program = Cl.CreateProgramWithSource(Context, 1, new string[] { programSource }, null, out status);
            CheckStatus(status);

            // build the program
            status = Cl.BuildProgram(program, 1, new[] { Device }, "", null, IntPtr.Zero);
            if (status != ErrorCode.Success)
            {
                InfoBuffer infoBuffer = Cl.GetProgramBuildInfo(program, Device, ProgramBuildInfo.Log, out status);
                CheckStatus(status);
                Console.WriteLine("Build Error: " + infoBuffer.ToString());
                System.Environment.Exit(1);
            }

            // create the vector addition kernel
            OpenCL.Net.Kernel kernel = Cl.CreateKernel(program, kernelName, out status);
            CheckStatus(status);

            _kernelsToRelease.Add(kernel);


            CheckStatus(Cl.ReleaseProgram(program));
            return kernel;
        }

        public void Dispose()
        {
            CheckStatus(Cl.ReleaseCommandQueue(CommandQueue));
            CheckStatus(Cl.ReleaseContext(Context));

            foreach (var buffer in _buffersToRelease)
            {
                CheckStatus(Cl.ReleaseMemObject(buffer));
            }

            foreach (var kernel in _kernelsToRelease)
            {
                CheckStatus(Cl.ReleaseKernel(kernel));
            }
        }
    }
}
