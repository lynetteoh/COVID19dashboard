from django.apps import AppConfig
import pandas as pd
import sys


class DashboardConfig(AppConfig):
    name = 'dashboard'

    def ready(self):
        if 'runserver' not in sys.argv:
            return True
        from dashboard.models import Case, State, Country
        