CONTACT_RESPONSE_RULES = [
    {
        "name": "contact_with_label_and_phone",
        "modes": {"fast", "policy", "dorm", "grad"},
        "conditions": ("contact_intent", "has_label", "has_phone"),
        "template_key": "contact_with_label_and_phone",
        "priority": 10,
    },
    {
        "name": "contact_with_phone_only",
        "modes": {"fast", "policy", "dorm", "grad"},
        "conditions": ("contact_intent", "has_phone"),
        "template_key": "contact_with_phone_only",
        "priority": 20,
    },
    {
        "name": "contact_retry_full_name",
        "modes": {"fast", "policy", "dorm", "grad"},
        "conditions": ("contact_intent", "not_has_phone"),
        "template_key": "contact_retry_full_name",
        "priority": 30,
    },
    {
        "name": "label_with_url",
        "modes": {"fast"},
        "conditions": ("has_label", "has_url"),
        "template_key": "label_with_url",
        "priority": 40,
    },
]
