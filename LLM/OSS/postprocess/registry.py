MODE_PIPELINES = {
    "fast": {
        "processor": "sub_answer",
        "use_schedule_first": True,
        "use_dept_clarification": True,
        "rule_set": "contact",
    },
    "policy": {
        "processor": "policy",
        "rule_set": "contact",
    },
    "dorm": {
        "processor": "dorm",
        "rule_set": "contact",
    },
    "grad": {
        "processor": "grad",
        "rule_set": "contact",
    },
    "topic": {
        "processor": "topic",
        "rule_set": "topic",
    },
}
