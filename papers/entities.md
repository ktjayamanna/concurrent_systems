Entities (stateful or rule-enforcing)
- SpeculativeExecutionEngine
- SpeculativeAgentTrio
- Predictor (and predictor variants)
- HazardDetectionUnit
- ReorderBuffer

Fields (attach to entities)
- ToolCall / Tool
- SafeTool / UnsafeTool (classification label)
- SpeculativeDispatch (event/operation)
- WorkerToolPredictor (role inside Predictor or AgentTrio)
- WorkerToolExecutor (role inside execution pipeline)
- Evaluator (external role)
- ConfigFlag / ModeToggle
- Timeout
