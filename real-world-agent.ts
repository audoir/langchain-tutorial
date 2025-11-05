import "dotenv/config";
import * as z from "zod";
import { createAgent, tool, type Runtime } from "langchain";
import { MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";

// define system prompt
const systemPrompt = `You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.`;

// define tools
const getWeather = tool(({ city }) => `It's always sunny in ${city}!`, {
  name: "get_weather",
  description: "Get the weather for a given city",
  schema: z.object({
    city: z.string(),
  }),
});

// this tool uses the runtime context to get the user's location
type AgentRuntime = Runtime<{ user_id: string }>;
const getUserLocation = tool(
  (_, config) => {
    const { user_id } = (config as AgentRuntime).context;
    return user_id === "1" ? "Florida" : "SF";
  },
  {
    name: "get_user_location",
    description: "Retrieve user information based on user ID",
    schema: z.object({}),
  }
);

// define agent
const agent = createAgent({
  model: new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
    maxTokens: 1000,
    // timeout: 30, // there's a bug here, it causes the agent to hang
  }),
  tools: [getWeather, getUserLocation],
  systemPrompt,
  checkpointer: new MemorySaver(), // save agent state
  responseFormat: z.object({
    punny_response: z.string(),
    weather_conditions: z.string(),
  }), // define structured output
});

// this configures the agent to use a specific thread_id and user_id
const config = {
  configurable: { thread_id: "1" },
  context: { user_id: "1" },
};

// run agent with config
const response = await agent.invoke(
  {
    messages: [{ role: "user", content: "what is the weather outside?" }],
  },
  config
);
console.log(response.structuredResponse);

// continue conversation
const thankYouResponse = await agent.invoke(
  { messages: [{ role: "user", content: "thank you!" }] },
  config
);
console.log(thankYouResponse.structuredResponse);
