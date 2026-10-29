class ExperimentResult:
    def __init__(self, values):
        self.values = dict(values)

    def __eq__(self, other):
        if not isinstance(other, ExperimentResult):
            return NotImplemented
        return (self.values["run_id"], self.values["model"]) == (
            other.values["run_id"], other.values["model"])

    def __lt__(self, other):
        if not isinstance(other, ExperimentResult):
            return NotImplemented
        return self.values["total_seconds"] < other.values["total_seconds"]
