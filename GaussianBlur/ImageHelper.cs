using System;
using System.Drawing;
using OpenCL.Net;
using static System.Net.Mime.MediaTypeNames;

namespace GaussianBlur
{
    public static class ImageHelper
    {
        public static float4[] GetImageData(Bitmap image)
        {
            // https://stackoverflow.com/questions/18766004/how-to-convert-from-system-drawing-bitmap-to-grayscale-then-to-array-of-doubles
            int width = image.Width;
            int height = image.Height;

            float4[] imageData = new float4[width * height];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Color pixel = image.GetPixel(x, y);
                    imageData[y * width + x] = ColorToFloat4(pixel);
                }
            }

            return imageData;
        }


        public static Bitmap CreateImageFromData(float4[] imageData, int width, int height)
        {
            Bitmap image = new Bitmap(width, height);

            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    float4 pixelValue = imageData[y * width + x];

                    image.SetPixel(x, y, Float4ToColor(pixelValue));
                }
            }

            return image;
        }

        private static float4 ColorToFloat4(Color color)
        {
            float Val(byte colorChannel) => Math.Max(0f, Math.Min(colorChannel / 255f, 1f));
            return new float4(Val(color.A), Val(color.R), Val(color.G), Val(color.B));
        }

        private static Color Float4ToColor(float4 float4)
        {
            int Val(float fl) => Math.Max(0, Math.Min((int)(fl * 255), 255));
            return Color.FromArgb(Val(float4.s0), Val(float4.s1), Val(float4.s2), Val(float4.s3));
        }

    }
}
