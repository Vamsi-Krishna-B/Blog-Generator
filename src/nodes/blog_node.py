from src.states.blogstate import BlogState 
from langchain_core.messages import SystemMessage,HumanMessage
from src.states.blogstate import Blog



class BlogNode:
    """
    A class to represent the blog node 
    """
    
    def __init__(self,llm):
        self.llm = llm
        
    def title_creation(self,state:BlogState):
        """
        Create the title for the blog 
        """
        
        if "topic" in state and state["topic"]:
            prompt = """
                    You are an expert blog content writer. Use Markdown formatting. Generate a Blog 
                    title for the {topic}.This title shoud be creative and SEO friendly.Give only the title in 
                    a single line.
            """ 
            system_message = prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog":{"title":response.content}}
        
    def content_generation(self,state:BlogState):
        """
        Create the content for the blog 
        """
        if "topic" in state and state["topic"]:
            prompt = """
                    You are an expert blog content writer. Use Markdown formatting. Generate a detailed breakdown Blog 
                    content for the {topic} .This content shoud be creative and SEO friendly.
            """ 
            system_message = prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog":{"title":state["blog"]["title"],"content":response.content}}
    
    def translation(self,state:BlogState):
        """
        Translate the content to specified languages 
        """
        translation_prompt = """
        Translate the following content into {current_language}.
        - Maintain the original tone, style, and formatting.
        - Adapt cultural references and idioms to be appropriate for {current_language}.
        
        ORIGINAL CONTENT:
        {blog_content} 
        """
        blog_content = state["blog"]["content"]
        message = [
            HumanMessage(translation_prompt.format(current_language=state["current_language"],blog_content=blog_content))
        ]
        translation_content = self.llm.with_structured_output(Blog).invoke(message)

        return {"blog":translation_content}
        
    def route(self,state:BlogState):
        return {"current_language":state["current_language"]}
    
    def route_decision(self,state:BlogState):
        """
        Route the content to specified languages 
        """
        if state["current_language"]=="hindi":
            return "hindi"
        elif state["current_language"]=="french":
            return "french"
        elif state["current_language"]=="spain":
            return "spain"
        else:
            return state["current_language"]
            
        