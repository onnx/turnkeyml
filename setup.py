from setuptools import setup

with open("src/turnkeyml/version.py", encoding="utf-8") as fp:
    version = fp.read().split('"')[1]

setup(
    name="turnkeyml",
    version=version,
    description="TurnkeyML Tools and Models",
    author="Jeremy Fowers, Daniel Holanda, Ramakrishnan Sivakumar, Victoria Godsoe",
    author_email="turnkeyml@amd.com",
    package_dir={"": "src", "turnkeyml_models": "models"},
    packages=[
        "turnkeyml",
        "turnkeyml.tools",
        "turnkeyml.tools.discovery",
        "turnkeyml.sequence",
        "turnkeyml.cli",
        "turnkeyml.common",
        "turnkeyml.llm",
        "turnkeyml.llm.tools",
        "turnkeyml.llm.tools.ort_genai",
        "turnkeyml.llm.tools.ryzenai_npu",
        "turnkeyml_models",
        "turnkeyml_models.graph_convolutions",
        "turnkeyml_models.selftest",
        "turnkeyml_models.timm",
        "turnkeyml_models.torch_hub",
        "turnkeyml_models.torchvision",
        "turnkeyml_models.transformers",
    ],
    install_requires=[
        "invoke>=2.0.0",
        # 1.16.2 causes a crash:
        #   ImportError: DLL load failed while importing onnx_cpp2py_export
        "onnx>=1.11.0,<1.16.2",
        "onnxmltools==1.10.0",
        "onnxruntime-directml>=1.10.1",
        "torch>=1.12.1",
        "pyyaml>=5.4",
        "typeguard>=2.3.13",
        "packaging>=20.9",
        # Necessary until upstream packages account for the breaking
        # change to numpy
        "numpy<2.0.0",
        "pandas>=1.5.3",
        "fasteners",
        "GitPython>=3.1.40",
        "psutil",
    ],
    extras_require={
        "llm": [
            "tqdm",
            "torch>=2.0.0",
            "transformers",
            "accelerate",
            "py-cpuinfo",
            "sentencepiece",
            "datasets",
            "fastapi",
            "uvicorn[standard]",
        ],
        "llm-oga-dml": [
            "onnxruntime-directml==1.19.0",
            "onnxruntime-genai-directml==0.4.0",
            "tqdm",
            "torch>=2.0.0",
            "transformers",
            "accelerate",
            "py-cpuinfo",
            "sentencepiece",
            "datasets",
            "fastapi",
            "uvicorn[standard]",
        ],
        "llm-oga-npu": [
            "transformers",
            "torch",
            "onnx==1.16.0",
            "onnxruntime==1.18.0",
            "numpy==1.26.4",
            "tqdm",
            "accelerate",
            "py-cpuinfo",
            "sentencepiece",
            "datasets",
            "fastapi",
            "uvicorn[standard]",
        ],
    },
    classifiers=[],
    entry_points={
        "console_scripts": [
            "turnkey=turnkeyml:turnkeycli",
            "turnkey-llm=turnkeyml.llm:lemonadecli",
            "lemonade=turnkeyml.llm:lemonadecli",
        ]
    },
    python_requires=">=3.8, <3.12",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={
        "turnkeyml_models": ["requirements.txt", "readme.md"],
    },
)
