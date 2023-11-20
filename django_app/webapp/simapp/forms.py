from django import forms

from .. import SPPy


class SimulationVariables(forms.Form):
    lst_parameter_name: list = [
        ("abc", "abc")
    ]

    parameter_name = forms.ChoiceField(label="Parameter Name", choices=lst_parameter_name)
    battery_cell_model = forms.CharField(max_length=200)
    solver_type = forms.CharField(max_length=200)
    cycler = forms.CharField(max_length=200)

print(SPPy.battery_components.parameter_set_manager.ParameterSets.list_parameters_sets())
