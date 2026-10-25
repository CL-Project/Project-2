"""Verify the environment: Python version, GPU/driver visibility, key library versions."""
import sys

def main():
    print("Python:", sys.version.split()[0])
    try:
        import torch
        print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except ImportError:
        print("torch not installed")
    for lib in ["gensim", "transformers", "sentence_transformers", "numpy", "scipy", "pandas"]:
        try:
            m = __import__(lib)
            print(f"{lib}:", getattr(m, "__version__", "ok"))
        except ImportError:
            print(f"{lib}: MISSING")

if __name__ == "__main__":
    main()
