import "dotenv/config";
import * as z from "zod";
import { createAgent, tool, type Runtime } from "langchain";
import { MemorySaver } from "@langchain/langgraph";

const systemPrompt = `You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.`;

const getWeather = tool(({ city }) => `It's always sunny in ${city}!`, {
  name: "get_weather",
  description: "Get the weather for a given city",
  schema: z.object({
    city: z.string(),
  }),
});

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

const responseFormat = z.object({
  punny_response: z.string(),
  weather_conditions: z.string(),
});

const checkpointer = new MemorySaver();
const agent = createAgent({
  model: "gpt-4o",
  tools: [getWeather, getUserLocation],
  systemPrompt,
  checkpointer,
  responseFormat,
});

const config = {
  configurable: { thread_id: "1" },
  context: { user_id: "1" },
};

const response = await agent.invoke(
  {
    messages: [{ role: "user", content: "what is the weather outside?" }],
  },
  config
);
console.log(response.structuredResponse);

const thankYouResponse = await agent.invoke(
  { messages: [{ role: "user", content: "thank you!" }] },
  config
);
console.log(thankYouResponse.structuredResponse);
