Entities (stateful or rule-enforcing)
- SpeculativeExecutionEngine
- Predictor (and predictor variants)
- HazardDetectionUnit
- ReorderBuffer

Fields (attach to entities)
- ToolCall / Tool
- SafeTool / UnsafeTool (classification label)
- SpeculativeDispatch (event/operation)
- WorkerToolPredictor (role inside Predictor)
- WorkerToolExecutor (role inside execution pipeline)
- Evaluator (external role)
- ConfigFlag / ModeToggle
- Timeout
