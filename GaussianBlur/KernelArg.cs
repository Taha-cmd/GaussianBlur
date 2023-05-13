using System;
using Microsoft.Win32;
using OpenCL.Net;

namespace GaussianBlur
{
    public class KernelArg
    {
        public object Value { get; }
        public IntPtr Size { get; }

        private KernelArg(object value, IntPtr size)
        {
            Value = value;
            Size = size;
        }

        public static KernelArg Get<T>(T value) where T : unmanaged
        {
            unsafe
            {
                return new KernelArg(value, (IntPtr)sizeof(T));
            }
        }

        public static KernelArg Get<T>(IMem<T> value) where T : unmanaged
        {
            unsafe
            {
                return new KernelArg(value, (IntPtr)sizeof(IntPtr));
            }
        }

    }
}
