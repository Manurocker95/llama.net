using LLaMA.NET.Native;
using LLaMA.NET.LibLoader;
using System;

namespace LLaMA.NET
{
    /// <summary>
    /// A factory for LLaMA models. Processed with <seealso cref="WhisperProcessor"/>
    /// </summary>
    public class LLaMAModel : System.IDisposable
    {
        public IntPtr ctx;
        private bool isDisposed = false;
        private bool isFromFile = false;
        private bool isFromBytes = false;

        private LLaMAModel(IntPtr context)
        {
            this.ctx = context;
        }

        /// <summary>
        /// Creates a new LLaMAModelFactory from a model path.
        /// </summary>
        /// <param name="modelPath">The path to the model.</param>
        public static LLaMAModel FromPath(string modelPath)
        {
            LibLoader.LibLoader.LibraryLoad();

            return new LLaMAModel(
                LLaMANativeMethods.llama_init_from_file(
                    modelPath, 
                    LLaMANativeMethods.llama_context_default_params()
                )
            );
        }

        /// <summary>
        /// Creates a LLaMARunner for this model.
        /// </summary>
        /// <returns>A new LLaMARunner.</returns>
        public LLaMARunner CreateRunner()
        {
            return new LLaMARunner(this);
        }

        public void Dispose()
        {
            // If already disposed, do nothing.
            if (isDisposed)
            {
                return;
            }

            // Dispose of unmanaged resources.
            if (ctx != null && ctx != IntPtr.Zero)
            {
                LLaMANativeMethods.llama_free(ctx);
            }

            isDisposed = true;
        }
    }
}