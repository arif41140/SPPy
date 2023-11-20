from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from bokeh.plotting import figure
from bokeh.embed import components

from .forms import SimulationVariables


def index(request) -> HttpResponse:
    if request.method == "POST":
        form = SimulationVariables(request.POST)
        if form.is_valid():
            HttpResponseRedirect('/result/')
    else:
        form = SimulationVariables()

    return render(request=request, template_name='input_simulation_variables.html', context={'form': form})


def result(request) -> HttpResponse:
    get_simulation_inputs(request=request)
    plot = figure()
    plot.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)

    script, div = components(plot)

    return render(request=request, template_name='result.html', context={'script': script, 'div': div})


def get_simulation_inputs(request):
    parameter_name = request.POST.get('parameter_name')
    battery_cell_model = request.POST.get('battery_cell_model')
    solver_type = request.POST.get('solver_type')
    cycler = request.POST.get('cycler')
    print(parameter_name, battery_cell_model, solver_type, cycler)
