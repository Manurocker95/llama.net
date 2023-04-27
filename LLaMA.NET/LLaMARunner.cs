using System.Runtime.InteropServices;
using System.Runtime;
using LLaMA.NET.Native;
using llama_token = System.Int32;
using System;
using System.Linq;

namespace LLaMA.NET
{
    public class LLaMARunner : IDisposable
    {
        private LLaMAModel _model;
        private string _prompt = "";
        private int _N_THREADS = 8;
        private llama_token[] _embeds = {};

        public LLaMARunner(LLaMAModel model)
        {
            this._model = model;
        }

        public LLaMARunner WithPrompt(string prompt)
        {
            this._prompt = prompt;

            // Convert prompt to embeddings
            var inputEmbeds = new llama_token[prompt.Length + 1];
            var inputTokenLength = LLaMANativeMethods.llama_tokenize(_model.ctx, prompt, inputEmbeds, inputEmbeds.Length, true);
            Array.Resize(ref inputEmbeds, inputTokenLength);
            
            // Evaluate the prompt
            for (int i = 0; i < inputEmbeds.Length; i++)
            {
                // batch size 1
                LLaMANativeMethods.llama_eval(_model.ctx, new llama_token[] { inputEmbeds[i] }, 1, this._embeds.Length + i, _N_THREADS);
            }

            // Add it and pass it along 😋
            this._embeds = this._embeds.Concat(inputEmbeds).ToArray();

            return this;
        }

        public LLaMARunner WithThreads(int nThreads)
        {
            this._N_THREADS = nThreads;
            return this;
        }

        public string Infer(out bool isEos, int nTokensToPredict = 50)
        {
            isEos = false;
            var prediction = "";

            for (int i = 0; i < nTokensToPredict; i++)
            {
                // Grab the next token
                var id = LLaMANativeMethods.llama_sample_top_p_top_k(_model.ctx, null, 0, 40, 0.8f, 0.2f, 1f / 0.85f);

                // Check if EOS, and break if otherwise!
                if (id == LLaMANativeMethods.llama_token_eos())
                {
                    isEos = true;
                    break;
                }

                // Add it to the context (all tokens, prompt + predict)
                var newEmbds = new llama_token[] { id };
                this._embeds = this._embeds.Concat(newEmbds).ToArray();

                // Get res!
                var res = Marshal.PtrToStringAnsi(LLaMANativeMethods.llama_token_to_str(_model.ctx, id));

                // Add to string
                prediction += res;

                // eval next token
                LLaMANativeMethods.llama_eval(_model.ctx, newEmbds, 1, this._embeds.Length, _N_THREADS);
            }

            return prediction;
        }

        public void Clear()
        {
            this._embeds = new llama_token[] {};
        }

        public void Dispose()
        {
        }
    }
}