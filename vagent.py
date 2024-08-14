import os
import requests
import chainlit as cl
from datetime import date, datetime, timedelta
from typing import Annotated, Optional, Union, Literal
from typing_extensions import TypedDict
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from dotenv import load_dotenv
import uuid
load_dotenv()


# User database
USER_DB = {
    "Anissa Putri": {"id": "1", "dob": "1990-05-15"},
    "Bimo Satrio": {"id": "2", "dob": "1995-11-22"},
    "Cecep Nugraha": {"id": "3", "dob": "2000-03-07"},
    "Tunggal Putra": {"id": "4", "dob": "2001-04-03"},
    "Allen Ganda": {"id": "5", "dob": "2001-04-03"},
}

CITY_CODES = {
    "jakarta": "CGK",
    "bandung": "BDO",
    "bali": "DPS",
    "denpasar": "DPS",
    "surabaya": "SUB",
    "jogjakarta": "YIA",
    "yogyakarta": "YIA",
    "solo": "SOC",
}

# Booking database
BOOKING_DB = {
    "flights": {
        1: {
            "user_id": "1",
            "flight_details": {"id": 1, "departure": "CGK", "arrival": "BDO", "date": "2024-08-16", "price": 110},
            "booking_time": "2024-03-15T10:30:00"
        }
    },
    "hotels": {
        2: {
            "user_id": "1",
            "hotel_details": {"id": 2, "name": "Grand Hyatt", "location": "BDO", "price": 180},
            "booking_time": "2024-03-15T11:00:00"
        }
    }
}

# TOOLS: Flights
@tool
def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    departure_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """Search for flights based on departure airport, arrival airport, and departure date."""
    flights = [
        {"id": 1, "departure": "CGK", "arrival": "DPS", "date": "2024-08-13", "price": 150},
        {"id": 2, "departure": "CGK", "arrival": "SUB", "date": "2024-08-13", "price": 120},
        {"id": 3, "departure": "CGK", "arrival": "TKG", "date": "2024-08-14", "price": 90},
        {"id": 4, "departure": "CGK", "arrival": "DPS", "date": "2024-08-14", "price": 180},
        {"id": 5, "departure": "CGK", "arrival": "DPS", "date": "2024-08-15", "price": 200},
        {"id": 6, "departure": "CGK", "arrival": "YIA", "date": "2024-08-15", "price": 130},
        {"id": 7, "departure": "CGK", "arrival": "BDO", "date": "2024-08-16", "price": 110},
        {"id": 8, "departure": "CGK", "arrival": "YIA", "date": "2024-08-16", "price": 160},
        {"id": 9, "departure": "CGK", "arrival": "DPS", "date": "2024-08-17", "price": 140},
        {"id": 10, "departure": "CGK", "arrival": "KNO", "date": "2024-08-17", "price": 170},
        {"id": 11, "departure": "CGK", "arrival": "SOC", "date": "2024-08-18", "price": 190},
        {"id": 12, "departure": "CGK", "arrival": "SUB", "date": "2024-08-18", "price": 150},
        {"id": 13, "departure": "CGK", "arrival": "YIA", "date": "2024-08-19", "price": 100},
        {"id": 14, "departure": "CGK", "arrival": "BDO", "date": "2024-08-19", "price": 140},
        {"id": 15, "departure": "CGK", "arrival": "SOC", "date": "2024-08-20", "price": 120},
        {"id": 16, "departure": "CGK", "arrival": "YIA", "date": "2024-08-20", "price": 150},
        {"id": 17, "departure": "CGK", "arrival": "BDO", "date": "2024-08-21", "price": 130},
        {"id": 18, "departure": "CGK", "arrival": "SUB", "date": "2024-08-21", "price": 160},
        {"id": 19, "departure": "CGK", "arrival": "YIA", "date": "2024-08-22", "price": 200},
        {"id": 20, "departure": "CGK", "arrival": "DPS", "date": "2024-08-22", "price": 180},
    ]
    
    if departure_airport:
        departure_airport = CITY_CODES.get(departure_airport.lower(), departure_airport.upper())
    if arrival_airport:
        arrival_airport = CITY_CODES.get(arrival_airport.lower(), arrival_airport.upper())

    results = []
    # print(">>>>>>>>>>>>> searching for flights...")
    # print(">>>>>>>>>>>>> departure_airport:", departure_airport)
    # print(">>>>>>>>>>>>> arrival_airport:", arrival_airport)
    # print(">>>>>>>>>>>>> departure_date:", departure_date)
    for flight in flights:
        if (not departure_airport or flight["departure"] == departure_airport) and \
           (not arrival_airport or flight["arrival"] == arrival_airport) and \
           (not departure_date or flight["date"] == str(departure_date)):
            results.append(flight)
    
    if not results:
        return f"No flights found for departure: {departure_airport}, arrival: {arrival_airport}, date: {departure_date}. Try expanding your search criteria."
    return results

@tool
def book_flight(flight_id: int) -> str:
    """Book a flight by its ID."""
    # print(">>>>>>>>>>>>> booking for flight...")
    flight = next((f for f in search_flights() if f["id"] == flight_id), None)
    if not flight:
        return f"Flight {flight_id} not found."
    
    if flight_id in BOOKING_DB["flights"]:
        return f"Flight {flight_id} is already booked."
    
    user_id = cl.user_session.get("user_profile", {}).get("id", "unknown")
    BOOKING_DB["flights"][flight_id] = {
        "user_id": user_id,
        "flight_details": flight,
        "booking_time": datetime.now().isoformat()
    }
    return f"Flight {flight_id} successfully booked for user {user_id}."

# TOOLS: Hotels
@tool
def search_hotels(
    location: Optional[str] = None,
    check_in_date: Optional[Union[datetime, date]] = None,
    check_out_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """Search for hotels based on location and dates."""
    # print(">>>>>>>>>>>>> searching for hotels...")
    hotels = [
        {"id": 1, "name": "Hotel Mulia", "location": "CGK", "price": 250},
        {"id": 2, "name": "Grand Hyatt", "location": "BDO", "price": 180},
        {"id": 3, "name": "Ayana Resort", "location": "DPS", "price": 320},
        {"id": 4, "name": "JW Marriott", "location": "SUB", "price": 270},
        {"id": 5, "name": "Plataran Resort", "location": "YIA", "price": 220},
        {"id": 6, "name": "The Ritz-Carlton", "location": "CGK", "price": 300},
        {"id": 7, "name": "Sheraton Hotel", "location": "BDO", "price": 160},
        {"id": 8, "name": "InterContinental", "location": "DPS", "price": 290},
        {"id": 9, "name": "Bumi Surabaya", "location": "SUB", "price": 240},
        {"id": 10, "name": "Royal Ambarrukmo", "location": "YIA", "price": 210},
        {"id": 11, "name": "Pullman Hotel", "location": "CGK", "price": 230},
        {"id": 12, "name": "Aston Pasteur", "location": "BDO", "price": 170},
        {"id": 13, "name": "Alila Villas", "location": "DPS", "price": 350},
        {"id": 14, "name": "Hotel Majapahit", "location": "SUB", "price": 260},
        {"id": 15, "name": "Sahid Jaya Hotel", "location": "YIA", "price": 200},
        {"id": 16, "name": "The Dharmawangsa", "location": "CGK", "price": 280},
        {"id": 17, "name": "Hotel Santika", "location": "BDO", "price": 150},
        {"id": 18, "name": "W Bali", "location": "DPS", "price": 400},
        {"id": 19, "name": "Garden Palace", "location": "SUB", "price": 220},
        {"id": 20, "name": "Hyatt Regency", "location": "YIA", "price": 240}
    ]
    
    if location:
        location = CITY_CODES.get(location.lower(), location.upper())

    results = []
    for hotel in hotels:
        if not location or hotel["location"] == location:
            results.append(hotel)
    
    if not results:
        return "No hotels found. Try expanding your search criteria."
    return results

@tool
def book_hotel(hotel_id: int) -> str:
    """Book a hotel by its ID."""
    # print(">>>>>>>>>>>>> booking for hotel...")
    hotel = next((h for h in search_hotels() if h["id"] == hotel_id), None)
    if not hotel:
        return f"Hotel {hotel_id} not found."
    
    if hotel_id in BOOKING_DB["hotels"]:
        return f"Hotel {hotel_id} is already booked."
    
    user_id = cl.user_session.get("user_profile", {}).get("id", "unknown")
    BOOKING_DB["hotels"][hotel_id] = {
        "user_id": user_id,
        "hotel_details": hotel,
        "booking_time": datetime.now().isoformat()
    }
    return f"Hotel {hotel_id} successfully booked for user {user_id}."


# TOOLS: Image generator
@tool
def generate_image(image_description: str) -> str:
    """Generate an image based on the given description using DALL-E."""
    # print(">>>>>>>>>>>>> generating image...")
    img_prompt = PromptTemplate(
        input_variables=["image_desc"],
        template="Generate a detailed prompt of length 100 characters or less to generate an image based on the following description: {image_desc}",
    )
    chain = LLMChain(llm=llm, prompt=img_prompt)
    image_url = DallEAPIWrapper(
        # model='dall-e-2',
        model='dall-e-3',
        # quality='standard',
        quality='hd',
        model_kwargs={'style': 'vivid'},
        # model_kwargs={'style': 'nautral'},
    ).run(chain.run(image_description))
    
    response = requests.get(image_url)
    if response.status_code == 200:
        os.makedirs("generated_images", exist_ok=True)
        local_path = f"generated_images/dalle_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        with open(local_path, "wb") as file:
            file.write(response.content)
        return f"Image generated and saved locally: {local_path}"
    else:
        return "Failed to generate and download the image."

# TOOLS: User profile
@tool
def get_user_info(user_id: str) -> dict:
    """Retrieve user information by user ID."""
    # print(">>>>>>>>>>>>>> getting user info...")
    for user_name, user_data in USER_DB.items():
        if user_data["id"] == user_id:
            return {"name": user_name, **user_data}
    return f"User with ID {user_id} not found."

@tool
def get_user_bookings(user_id: str) -> dict:
    """Retrieve active bookings for a user."""
    # print(">>>>>>>>>>>>>> getting user bookings...")
    bookings = {
        "flights": [],
        "hotels": []
    }
    for flight_id, flight_booking in BOOKING_DB["flights"].items():
        if flight_booking["user_id"] == user_id:
            bookings["flights"].append(flight_booking["flight_details"])
    for hotel_id, hotel_booking in BOOKING_DB["hotels"].items():
        if hotel_booking["user_id"] == user_id:
            bookings["hotels"].append(hotel_booking["hotel_details"])
    return bookings

# Utility functions
def handle_tool_error(state) -> dict:
    # print(">>>>>>>>>>>>>> handling tool error...")
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

# State
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: dict

# Assistant
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            prompt_input = {
                "messages": state["messages"],
                "user_id": state["user_info"]["id"]
            }
            result = self.runnable.invoke(prompt_input, config)
            # Reprompt if LLM returns an empty response
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# LLM and prompt
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful vacation planning assistant. "
            "Use cheerful, friendly and informative tone in your every answer. "
            "Your task is to help the user plan their vacation. "
            "Start with arranging detailed and thorough itinerary before booking any activity or accomodation. "
            "If needed, use the provided tools to find recent information online or search for flights and hotels to assist the user's queries. "
            "If a search comes up empty, expand your search before giving up. "
            "\n\nCurrent user ID: {user_id}\n"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

safe_tools = [
    TavilySearchResults(max_results=5, search_depth='advanced'),
    search_flights,
    search_hotels,
    get_user_info,
    get_user_bookings,
    generate_image,
]

sensitive_tools = [
    book_flight,
    book_hotel,
]
sensitive_tool_names = {t.name for t in sensitive_tools}

assistant_runnable = (
    assistant_prompt
    | llm.bind_tools(safe_tools + sensitive_tools)
).with_config({"run_name": "Assistant"})

# Graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("safe_tools", create_tool_node_with_fallback(safe_tools))
builder.add_node("sensitive_tools", create_tool_node_with_fallback(sensitive_tools))
builder.add_edge(START, "assistant")

def route_tools(state: State) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
    # print(">>>>>>>>>>>>>> routing tools...")
    next_node = tools_condition(state)
    if next_node == END:
        return END
    ai_message = state["messages"][-1]
    if not hasattr(ai_message, 'tool_calls') or not ai_message.tool_calls:
        return "safe_tools"
    first_tool_call = ai_message.tool_calls[0]
    tool_name = first_tool_call.name if hasattr(first_tool_call, 'name') else first_tool_call.get("name")
    
    if tool_name in sensitive_tool_names:
        return "sensitive_tools"
    return "safe_tools"

builder.add_conditional_edges("assistant", route_tools)
builder.add_edge("safe_tools", "assistant")
builder.add_edge("sensitive_tools", "assistant")

memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["sensitive_tools"],
)

def match_user_profile(name: str) -> dict:
    """Match user name with existing profiles or create a new one."""
    for user_name, user_data in USER_DB.items():
        if name.lower() in user_name.lower():
            return {"name": user_name, **user_data}
    
    new_id = str(len(USER_DB) + 1)
    new_profile = {"id": new_id, "dob": "Unknown"} 
    USER_DB[name.title()] = new_profile
    return {"name": name.title(), **new_profile}



@cl.on_chat_start
async def setup():
    thread_id = str(uuid.uuid4())
    cl.user_session.set("user_profile", None)
    cl.user_session.set("thread_id", thread_id)
    await cl.Message("🌴 Hi there! I'm your vacation planning assistant. I need your name to get started—what's your name? 😊").send()

@cl.on_message
async def main(message: cl.Message):
    user_profile = cl.user_session.get("user_profile")
    thread_id = cl.user_session.get("thread_id")

    if user_profile is None:
        user_profile = match_user_profile(message.content)
        cl.user_session.set("user_profile", user_profile)

        user_info = get_user_info(user_profile["id"])
        initial_greeting = f"Hello, {user_info['name'].title()}! How can I assist you with your vacation planning today?"
        await cl.Message(initial_greeting).send()
    else:
        config = {
            "configurable": {
                "user_id": user_profile["id"],
                "thread_id": thread_id,
            }
        }

        events = graph.stream(
            {
                "messages": [("user", message.content)],
                "user_info": user_profile
            },
            config,
            stream_mode="values"
        )

        final_response = None
        for event in events:
            # print(f">>>>>>>>>>>>>> Event: {event}")
            if event.get("messages"):
                final_response = event["messages"][-1]

                if hasattr(final_response, 'tool_calls') and final_response.tool_calls:
                    for tool_call in final_response.tool_calls:
                        # print(f">>>>>>>>>>>>>> Tool call: {tool_call}")
                        if isinstance(tool_call, dict):
                            tool_name = tool_call.get('name')
                            tool_args = tool_call.get('args', {})
                        else:
                            tool_name = tool_call.name if hasattr(tool_call, 'name') else None
                            tool_args = tool_call.args if hasattr(tool_call, 'args') else {}
                        
                        # print(f">>>>>>>>>>>>>> Tool name: {tool_name}")
                        # print(f">>>>>>>>>>>>>> Tool args: {tool_args}")
                        
                        if tool_name == 'book_flight':
                            flight_id = tool_args.get('flight_id')
                            if flight_id:
                                booking_result = book_flight(flight_id)
                                # print(f">>>>>>>>>>>>>> Booking result: {booking_result}")
                                # Add the booking result to the messages
                                event["messages"].append(ToolMessage(content=booking_result, name='book_flight'))
                            else:
                                print(">>>>>>>>>>>>>> Error: No flight_id found in tool arguments")
        if final_response:
            # print(f">>>>>>>>>>>>>> Final response: {final_response}")
            content = final_response.content

            # Handle tool calls
            if hasattr(final_response, 'tool_calls') and final_response.tool_calls:
                for tool_call in final_response.tool_calls:
                    # print(f">>>>>>>>>>>>>> Tool call: {tool_call}")
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get('name')
                        tool_args = tool_call.get('args', {})
                    else:
                        tool_name = tool_call.name if hasattr(tool_call, 'name') else None
                        tool_args = tool_call.args if hasattr(tool_call, 'args') else {}
                    
                    # print(f">>>>>>>>>>>>>> Tool name: {tool_name}")
                    # print(f">>>>>>>>>>>>>> Tool args: {tool_args}")
                    
                    if tool_name == 'book_flight':
                        flight_id = tool_args.get('flight_id')
                        if flight_id:
                            booking_result = book_flight(flight_id)
                            # print(f">>>>>>>>>>>>>> Booking result: {booking_result}")
                            # Add the booking result to the messages
                            event["messages"].append(ToolMessage(content=booking_result, name='book_flight'))
                        else:
                            print(">>>>>>>>>>>>>> Error: No flight_id found in tool arguments")


            generated_images_dir = "generated_images"
            if os.path.exists(generated_images_dir):
                image_files = [f for f in os.listdir(generated_images_dir) if f.endswith('.png')]
                if image_files:
                    # Sort files by creation time and get the most recent one
                    latest_image = max(image_files, key=lambda f: os.path.getctime(os.path.join(generated_images_dir, f)))
                    image_path = os.path.join(generated_images_dir, latest_image)
                    
                    # Check if the image was created in the last 5 seconds
                    creation_time = datetime.fromtimestamp(os.path.getctime(image_path))
                    if datetime.now() - creation_time < timedelta(seconds=5):
                        image_file = cl.Image(name=latest_image, path=image_path)
                        await cl.Message(content="", elements=[image_file]).send()

            if content.strip():
                await cl.Message(content=content).send()
            else:
                # If content is empty, check for tool messages
                tool_messages = [msg for msg in event.get("messages", []) if isinstance(msg, ToolMessage)]
                if tool_messages:
                    content = "\n".join([msg.content for msg in tool_messages])
                    await cl.Message(content=content).send()
                else:
                    await cl.Message(content="I'm sorry, but I couldn't process your request. Can you please try again?").send()



