# from multiprocessing import context
# import langchain
from langchain_core.documents import Document
import os
doc = Document(
    page_content="this is the main text content I am using to create RAG",
    metadata={
        "source":"example.txt",
        "pages":1,
        "author":"stackashu",
        "date_created":"2025-01-01"
    }
)

# print(doc)

##create a simple txt file 
os.makedirs("./data/text_files",exist_ok=True)

samples = {
    "./data/text_files/ai_topic.txt": """Artificial intelligence (AI) is a broad discipline of computer science focused on creating systems capable of performing tasks that normally require human intelligence. 
The field encompasses a variety of sub-disciplines, including machine learning, natural language processing, computer vision, and robotics. 
Machine learning, a central subset of AI, involves developing algorithms that enable computers to recognize patterns and make predictions or decisions based on data. 
Natural language processing equips computers to understand and generate human language, enabling applications like chatbots and virtual assistants. 
Computer vision gives machines the ability to interpret and process visual information from the world, leading to breakthroughs in image and video analysis. 
AI's applications are transformative, with impacts spanning healthcare, finance, transportation, education, and entertainment. In healthcare, AI assists with disease diagnosis and personalized treatment plans. 
Financial institutions leverage AI for fraud detection, algorithmic trading, and customer service automation. Autonomous vehicles rely on AI to perceive their environment and make real-time driving decisions. 
As AI continues to advance, ethical considerations such as bias, interpretability, and the impact on jobs become increasingly important, driving ongoing research and debate within the field. 
Despite challenges, the promise of AI for improving efficiency, enabling innovation, and solving complex problems continues to drive its rapid evolution and adoption.
""",
    "./data/text_files/renewable_energy_topic.txt": """Renewable energy refers to power derived from resources that are naturally replenished, such as sunlight, wind, rain, tides, waves, geothermal heat, and biomass. 
Among these, solar and wind power have seen remarkable growth in recent years due to advances in technology and declining production costs. 
Solar panels convert sunlight directly into electricity, making it possible for homes and businesses to generate their own clean power. Wind turbines harness the kinetic energy of moving air, contributing significantly to the energy mix of many countries. 
Hydropower, one of the oldest renewable sources, relies on flowing water to generate electricity and remains a major source of power worldwide. 
Renewable energy has numerous benefits, including reducing greenhouse gas emissions, creating jobs, and enhancing energy security. 
By relying less on fossil fuels, societies can mitigate the harmful effects of climate change—such as rising sea levels, extreme weather events, and threats to biodiversity. 
Challenges persist, including variability in energy generation and the need for effective storage solutions, yet investments in grid infrastructure and battery technology are addressing these concerns. 
Transitioning to renewable energy is crucial for a sustainable future, offering a pathway to meet growing energy demands while preserving the planet for future generations.
"""
}
for filePath, content in samples.items():
    with open(filePath,'w',encoding='utf-8') as f:
        f.write(content)


### TextLoader 
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/text_files/ai_topic.txt",encoding="utf-8")
document = loader.load()
print(document)

##Directory Loader 
from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader , PyMuPDFLoader

##load all the text files from the directory 
dir_loader = DirectoryLoader(
    "./data/pdf",
    glob="**/*.pdf" , ## pattern to match files,
    loader_cls = PyPDFLoader, ##loader class to use
    # loader_kwargs={'encoding':'utf-8'},
    show_progress=False
)

documents = dir_loader.load()
print(documents)