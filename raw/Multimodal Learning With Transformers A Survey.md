---
title: "Multimodal Learning With Transformers: A Survey"
source: "https://medium.com/@EleventhHourEnthusiast/multimodal-learning-with-transformers-a-survey-3b28b1dcaf03"
author:
  - "[[Eleventh Hour Enthusiast]]"
published: 2025-05-09
created: 2026-06-26
description: "Paper Review"
tags:
  - "clippings"
---
## Paper Review

**Introduction**

Modern AI systems increasingly need to process multiple types of data simultaneously, such as images with text, video with audio, or sensor data with natural language descriptions. Transformer architectures have emerged as a powerful solution for handling these multimodal tasks. As highlighted by Xu et al. (2023), transformers can effectively integrate diverse data modalities by treating them as nodes in a graph-like structure, enabling flexible, natural combinations of different information sources. This blog post explores three key architectural patterns for multimodal transformers — early fusion, cross-modal attention, and hierarchical attention — and examines how they combine information to achieve robust performance across a variety of tasks.

**What Are Multimodal Models?**

Multimodal learning refers to the ability of AI systems to understand and process different types of data at the same time. The term “multimodal” means using more than one way of understanding something. For example, data can come from the same source, such as images captured by different cameras (homogeneous modalities), or from different sources, like combining images with text (heterogeneous modalities).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*T6diTucWaZIb_b1uKqfIkg.png)

From a broader perspective, multimodal data involves integrating information from various senses, like sight, sound, touch, and smell, to create a complete representation of the environment. In terms of data, it means combining different types of information, such as images, numbers, text, symbols, sounds, time-based data, or other complex data forms.

**Historical Development of Multimodal Algorithms**

The survey paper outlines four distinct phases in the evolution of multimodal research.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*ctiqqrz8GAY2FayeL6BK9Q.png)

**Single Modality (1980–2000).** The first phase focused on basic computing capabilities. Statistical algorithms and image processing techniques were used for face recognition systems, laying the groundwork for early methods in this area. During this time, IBM research teams made important contributions to speech recognition using hidden Markov models (HMMs), which helped improve the accuracy and reliability of speech recognition technology. In the 1990s, Kanade’s team developed the Eigenfaces method for face recognition, using principal component analysis (PCA) to extract facial features and identify individuals based on statistical patterns in face images.

**Modality Conversion (2000–2010).** The second phase focused on improving human-computer interaction. The goal was to enable computers to simulate human behavior and make everyday tasks more convenient. Key developments included the AMI project in 2001, which aimed to use computers to record and process meeting data by analyzing audio, video, and text for better information retrieval and collaboration. In 2003, the CALO project introduced chatbot technology, which became the foundation for modern virtual assistants like Siri. In 2008, the social signal processing (SSP) project introduced the concept of social signal processing networks, which focused on understanding non-verbal cues, such as facial expressions, gestures, and voice tones, to interpret social interactions.

**Modality Fusion (2010–2020).** The third phase saw the integration of deep learning techniques and neural networks. A transformative multimodal deep learning algorithm developed by Ngiam in 2011 allowed for the fusion and analysis of different types of data, such as images and text, enabling joint learning of features from multiple sources. In 2012, a multimodal learning algorithm based on deep Boltzmann machines (DBMs) was introduced to model the dependencies and interactions between different modalities. By 2016, a neural image captioning algorithm with semantic attention revolutionized image processing by enabling automatic understanding and description of images.

**Large-scale Multimodal Models (2020 and beyond).** The fourth phase opened new opportunities for multimodal algorithms. The CLIP model, introduced in 2021, moved away from traditional fixed category labels and instead used image-text pairs to predict their similarity or generate new ones. Other notable models include DALL-E 2 (2022), which generates high-quality images from text prompts using a diffusion model and CLIP image embeddings; Microsoft’s BEiT-3 (2022), which pre-trains through masked data with a shared multiway transformer structure; KOSMOS-1 (2023), an advanced multimodal LLM capable of integrating information from various modalities; and PaLM-E (2023), which combines language and vision models to excel at both visual and language tasks.

**Technical Components of Multimodal Models**

According to the survey, multimodal large language models have several key technical components.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*vsfRIH5D99HdbnOh16h7MA.png)

**Knowledge Representation.** Knowledge representation involves converting text and images into forms that AI can process. For text, methods like Word2Vec were used in the past, but they had vocabulary limitations. To address this, subword tokenization methods, such as byte pair encoding, became more popular. Image tokenization is more complex and includes region-based, grid-based, and patch-based methods. Data from the METER model shows that improving the visual features of a model has a much greater impact on results than improving the text side, highlighting the importance of visual information.

**Learning Objectives Selection.** Common learning tasks for multimodal models include image-text contrast (ITC), masked language modeling (MLM), masked visual modeling (MVM), and image-text matching (TM). Combining different learning tasks can improve the performance of multimodal models. For example, the UNITER model uses multiple learning tasks like MLM and ITC, leading to strong performance in various scenarios. However, using too many tasks at once may not always improve results, as shown by experiments with the METER model.

**Model Structure Construction.** Depending on the model structure, multimodal models are divided into encoder-only models and encoder-decoder models. Encoder-only models use just the encoder part of the Transformer. Examples include CLIP and ALBEF, which are good for tasks like image-text retrieval but not for image captioning. Encoder-decoder models use both the encoder and decoder parts of the Transformer. Models like T5 and SimVLM use the decoder for tasks like text generation.

**Information Fusion.** Multimodal models can be divided into fusion encoder models and dual encoder models, depending on how they combine different types of data. The fusion encoder model allows the modalities to interact using self-attention or cross-attention operations. Fusion methods include single-stream and dual-stream approaches. The fusion encoder is used for tasks that require inference, while the dual encoder is better suited for tasks like retrieval.

**The Use of Prompts.** The prompt method helps bridge the gap between pretraining and fine-tuning in downstream tasks. Prompts are useful for handling tasks with little or no data, which has been widely adopted in large language models. For example, Visual ChatGPT uses a prompt manager to generate prompts that help ChatGPT understand and generate related images. In the CLIP model, prompts are used for zero-shot tasks, improving performance by generating informative prompts for text.

**Notable Multimodal Models and Their Applications**

The survey highlighted several important multimodal models.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*LZDpWpSqOGdp5f1UmVMGsg.png)

**Visual ChatGPT and MM-REACT.** Visual ChatGPT combines different visual foundation models (VFMs) to handle tasks like understanding and generating images. The system includes a Prompt Manager that helps manage interactions with VFMs, allowing them to provide feedback in an iterative way. By using prompts to introduce visual model information into ChatGPT, this system aligns visual features with text, improving ChatGPT’s ability to understand and generate visuals.

MM-REACT works similarly by combining ChatGPT with various visual models for multimodal tasks, especially visual question answering (VQA). When answering questions, ChatGPT uses visual models as tools, choosing which model to use based on the question being asked.

**BLIP-2 and LLaMA-Adapter.** BLIP-2 uses a method like Flamingo to encode images, using a Qformer model to extract features from the images. During training, both the visual encoder and the large language models (LLMs) are frozen, and only the Qformer is fine-tuned. The training happens in two stages: first, the Qformer and visual encoder are trained with standard multimodal tasks, and second, the Qformer-encoded features are added to the LLM for caption generation.

LLaMA-Adapter improves fine-tuning efficiency in LLaMA by adding adapters, which can be applied to multimodal tasks. For these tasks, images are encoded into multiscale feature vectors using a fixed visual encoder, and then the vectors are combined before being added to the adaptation prompt vectors.

**MiniGPT-4 and LLaVA.** MiniGPT-4 is a version of GPT-4, built by combining BLIP-2 and Vicuna. It transfers the Qformer and visual encoder from BLIP-2, freezes them along with the LLM, and only fine-tunes a linear layer on the visual side. This reduction in tunable parameters results in a much smaller model with only 15 million parameters.

LLaVA and MiniGPT-4 both focus on multimodal instruction fine-tuning but use different methods for data generation and training. LLaVA uses GPT-4 to create diverse fine-tuning data, which includes tasks like multi-turn question answering, image descriptions, and complex reasoning tasks.

**Applications of Multimodal Models**

Multimodal models have a wide range of applications across different fields, improving capabilities by combining various types of data.

**Image Captioning.** Image captioning involves generating short textual descriptions for given images. Models need to capture the semantic information of the images, detect key objects, actions, and features, and infer the relationships between objects. This application is particularly helpful for visually impaired users, enhancing their experience and engagement with the visual world.

**Text-to-Image Generation.** Text-to-image generation is one of the most popular applications of multimodal learning. Models such as OpenAI’s DALL-E 2 and Google’s Imagen have made significant advances in this area. By providing short textual descriptions as prompts, these models can generate novel images that accurately reflect the semantics of the text. These technologies offer new possibilities for creating and understanding images, bridging the gap between textual and visual communication.

**Sign Language Recognition.** Sign language recognition aims to recognize sign language gestures and convert them into text. It requires the model to align the temporal information of the visual (video frames) and audio modalities to identify the gestures and their corresponding spoken language. The RWTH PHOENIX Weather 2014T dataset, containing video recordings of German sign language from different signers, is commonly used for this task.

**Emotion Recognition and Video Processing.** While emotion recognition can be performed using only a single-modal dataset, performance can be improved by utilizing multimodal datasets as input. Multimodal inputs can take the form of video, text, and audio or can incorporate sensor data such as brainwave data. A real-world example is emotion recognition in music, where the model identifies the emotional content of music using audio features and lyrics.

In the domain of video and audio, multimodal fusion is also a growing trend. As image-text multimodal models transition to video-text and audio-text domains, a series of key models have emerged. For example, the VideoCoCa model for the image-text domain and the CLIP model led to the development of the VideoCLIP model.

**Challenges in Multimodal Learning**

Despite significant progress, the survey highlights several challenges in multimodal learning that still need to be addressed.

**Modality Expansion.** The sensors and data sources used in multimodal learning are diverse, offering rich information for more accurate analysis and recognition. For example, in emotion recognition, multiple modalities such as audio, facial expressions, electrocardiography (ECG), and electroencephalography (EEG) can be combined to gain a deeper understanding of emotional states. Audio can capture changes in speech tone and rate, visual data can analyze facial expressions and body language, and ECG and EEG signals provide physiological insights related to emotional changes.

**Time-Consuming Training.** Large models require significant computational resources. Due to their size, computations must often be distributed across clusters. In multi-user and multi-task environments, systems need to support multi-tenancy. Additionally, high reliability is essential, requiring models to be fault-tolerant and capable of adapting to dynamic changes. Combining multiple backbone models can further complicate the process.

**Lifelong/Continual Learning**: Traditional AI models are trained on specific datasets and then applied to tasks. This method, known as isolated learning, has the drawback of lacking memory capabilities. For real-world applications, multimodal large models need the ability to learn continuously, using past experiences to adapt and improve. This would allow models to build a more complex understanding of the world based on ongoing learning.

**Catastrophic Forgetting**: As AI moves toward artificial general intelligence (AGI), a challenge known as catastrophic forgetting arises. This occurs when a neural network, originally trained for one task, is repurposed for another and “forgets” its initial training. Recent research, such as BLIP-2, KOSMOS-1, BEiT-3, and PaLI, has explored two approaches to tackle this problem: first, avoiding catastrophic forgetting by using smaller networks and retraining with new data; second, using larger language networks as backbones to mitigate forgetting.

**Conclusion**

Multimodal LLMs represent a major advancement in AI research, allowing systems to process and integrate multiple types of data at the same time. According to the survey, these models have evolved through four key stages, starting from single-modality processing to large-scale multimodal integration. They now use advanced techniques for knowledge representation, learning objective selection, model construction, information fusion, and prompt usage.

Notable multimodal models have shown impressive capabilities in various applications, such as image captioning, text-to-image generation, sign language recognition, emotion recognition, and video processing. These applications highlight the potential of multimodal learning to make AI systems more versatile and human-like in understanding the world.

Despite the remarkable progress, the survey identifies several challenges that still need to be addressed, including modality expansion, time-consuming training, lifelong learning, and catastrophic forgetting. However, in the couple of years since the paper was published, many advances have been made in addressing these issues, including improvements in distributed training methods, more efficient fusion techniques, and better approaches to continual learning.

Reference:  
Wu, J., Gan, W., Chen, Z., Wan, S., & Yu, P. S. (2023). Multimodal Large Language Models: A Survey. *arXiv e-print, arXiv:2311.13165v1*.[  
https://doi.org/10.48550/arXiv.2311.13165](https://doi.org/10.48550/arXiv.2311.13165)