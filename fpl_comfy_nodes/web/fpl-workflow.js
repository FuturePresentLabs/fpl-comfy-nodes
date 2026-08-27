import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const WORKFLOW_URL = "/fpl/workflows/openrouter-image";
const LOAD_PARAM = "fpl_workflow";

function nodeTypes(graphData) {
  return new Set((graphData?.nodes ?? []).map((node) => node.type));
}

function hasFplBifrostNode(graphData) {
  return nodeTypes(graphData).has("FPLBifrostImageGeneration");
}

function looksLikeLocalDiffusionWorkflow(graphData) {
  const types = nodeTypes(graphData);
  if (hasFplBifrostNode(graphData)) {
    return false;
  }
  return (
    (graphData?.nodes?.length ?? 0) === 0 ||
    types.has("CheckpointLoaderSimple") ||
    types.has("KSampler") ||
    types.has("EmptyLatentImage") ||
    types.has("VAEDecode")
  );
}

async function fetchFplWorkflow() {
  const response = await api.fetchApi(WORKFLOW_URL);
  if (!response.ok) {
    throw new Error(`FPL workflow request failed: ${response.status}`);
  }
  return await response.json();
}

async function loadFplWorkflow(source = "fpl") {
  const workflow = await fetchFplWorkflow();
  await app.loadGraphData(workflow, true, true, "FPL OpenRouter Image", {
    openSource: source,
    skipAssetScans: true,
    silentAssetErrors: true,
  });
}

function shouldForceLoadFromUrl() {
  return new URLSearchParams(window.location.search).get(LOAD_PARAM) === "1";
}

function patchDefaultWorkflowLoader() {
  const originalLoadGraphData = app.loadGraphData.bind(app);
  app.loadGraphData = async function loadGraphDataWithFplDefault(graphData, ...args) {
    if (looksLikeLocalDiffusionWorkflow(graphData)) {
      try {
        graphData = await fetchFplWorkflow();
        args[2] ??= true;
        args[3] ??= "FPL OpenRouter Image";
        args[4] = {
          ...(args[4] ?? {}),
          openSource: args[4]?.openSource ?? "fpl-default",
          skipAssetScans: true,
          silentAssetErrors: true,
        };
      } catch (error) {
        console.error("Failed to replace default workflow with FPL workflow", error);
      }
    }
    return await originalLoadGraphData(graphData, ...args);
  };
}

app.registerExtension({
  name: "FPL.BifrostWorkflow",
  commands: [
    {
      id: "FPL_LoadBifrostImageWorkflow",
      label: "Load FPL Bifrost Image Workflow",
      function: () => loadFplWorkflow("fpl-command"),
    },
  ],
  async init() {
    patchDefaultWorkflowLoader();
  },
  async setup() {
    window.setTimeout(async () => {
      try {
        const graph = app.graph?.serialize?.();
        if (shouldForceLoadFromUrl() || looksLikeLocalDiffusionWorkflow(graph)) {
          await loadFplWorkflow("fpl-startup");
        }
      } catch (error) {
        console.error("Failed to load FPL Bifrost workflow", error);
      }
    }, 250);
  },
});
