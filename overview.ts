import "dotenv/config";
import * as z from "zod";
import { createAgent, tool } from "langchain";

// define tools
const getWeather = tool(({ city }) => `It's always sunny in ${city}!`, {
  name: "get_weather",
  description: "Get the weather for a given city",
  schema: z.object({
    city: z.string(),
  }),
});

// define agent
const agent = createAgent({
  model: "gpt-3.5-turbo",
  tools: [getWeather],
});

// run agent
console.log(
  await agent.invoke({
    messages: [{ role: "user", content: "What's the weather in Tokyo?" }],
  })
);
