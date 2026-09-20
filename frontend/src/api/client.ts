import createClient from "openapi-fetch";
import type { paths } from "./schema";

// Typed fetch client. `paths` is generated from the backend OpenAPI spec by
// `npm run gen:api` (openapi-typescript) → src/api/schema.d.ts. Every GET/POST is
// type-checked against the live Pydantic contract; the CI gate (gen:api:check)
// fails if the backend schema drifts from the committed types (F4).
//
// baseUrl "/" → same-origin: the Vite dev proxy forwards /api to the FastAPI
// service; in production the SPA is served by that same service.
export const api = createClient<paths>({ baseUrl: "/" });
