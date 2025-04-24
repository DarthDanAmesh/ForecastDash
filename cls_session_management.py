# Session State Management
class SessionState:
    def __init__(self):
        self.data = None
        self.models = {}
        self.forecasts = {}
        self.data_source = None
        self.connection_string = None
        self.uploaded_file = None
        self.model_cache = {}
        self.api_config = None
        self.data_fingerprints = {}
        self.accuracy_data = {}

    def reset(self):
        self.data = None
        self.models = {}
        self.forecasts = {}