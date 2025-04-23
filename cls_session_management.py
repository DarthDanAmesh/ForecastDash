# Session State Management
class SessionState:
    def __init__(self):
        self.data = None
        self.models = {}
        self.forecasts = {}
        self.data_source = None
        self.connection_string = None
        self.uploaded_file = None
        self.api_config = None

    def reset(self):
        self.data = None
        self.models = {}
        self.forecasts = {}