# Server API (OpenAPI)

The JSON endpoints of the [server](server.md), as an OpenAPI 3.1 spec:
[`openapi.json`](openapi.json) (also in the release archive, beside
`recommender.proto` for gRPC). Feed it to a client generator or an API
tool; the server's tests keep its field names in sync with the server.

<!-- Swagger UI is light-only: a light panel in the dark scheme. -->
<style>
  [data-md-color-scheme="slate"] #swagger-ui { background: #fff; border-radius: 4px; padding: 1px 1rem; }
</style>
<div id="swagger-ui"></div>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui.css">
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"></script>
<script>
  SwaggerUIBundle({
    url: new URL("../openapi.json", document.baseURI).href,
    dom_id: "#swagger-ui",
    // A docs page, not a console: no "Try it out" against a server.
    supportedSubmitMethods: [],
    defaultModelsExpandDepth: 0,
  });
</script>
