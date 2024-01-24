# LLMs_containers
**LLMs Singularity and Docker containers**

################################################################################
RAMS request ID  1026639

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force 
under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, 
conclusions or recommendations expressed in this material are those of the 
author(s) and do not necessarily reflect the views of the Department of the 
Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-ls basis
Delivered to the U.S. Government with Unlimited Rights, as defined in 
DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright 
notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 
or DFARS 252.227-7014 as detailed above. Use of this work other than as 
specifically authorized by the U.S. Government may violate any copyrights that 
exist in this work.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
################################################################################

**Summary**

  This software creates portable Docker and Singularity containers for 
Large Language Models (LLM): Falcon, LLAMA2, and MistralAI.  This enables 
running the LLMs on standalone computers, high-performance computers (HPC),
and cloud hosted platforms.

**Singularity**

**LLAMA2, Falcon, and MistralAI Models:**

  Squashfs models are on TX-Green on L4M_shared/Models

**To build:**

  singularity build llms.sif llms.def

  singularity build --sandbox llms_box llms.def			Note: builds Singularity sandbox

**To run:**

    singularity run --nv -B io/:/io/ -B <squashfs>:/io/hub/<model>:image-src=/ llms.sif <Your program details>

**Example LLM Python application:**

    singularity run --nv -B io/:/io/ -B Llama-2-7b-chat-hf.sqsh:/io/hub/models--meta-llama--Llama-2-7b-chat-hf:image-src=/ llms.sif python /io/llama2_cli.py "How to cook fish?"

**Jupyter notebook example:**

  singularity run --nv -B io/:/io/ -B falcon-7b.sqsh:/io/hub/models--tiiuae--falcon-7b:image-src=/ llms_box jupyter notebook --allow-root --ip='*' --NotebookApp.token='' --NotebookApp.password=''
**Docker**

**To build:**

  docker build . -t llms:latest 

**To run:**

  docker run -it llms:latest bash

**Docker LLM Python example:**

  docker run --gpus all -v /data/da23452/llm/llms/io:/io -v /data/da23452/llm/llama2/models--meta-llama--Llama-2-7b-chat-hf:/io/hub/models--meta-llama--Llama-2-7b-chat-hf llms:latest python llama2_cli.py "How to cook pasta?"

**Example Jupyter notebook using LangChain:**

  LangChain_example.ipynb

