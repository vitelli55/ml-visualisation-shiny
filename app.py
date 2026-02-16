import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models import gradient_descent

from shiny import App, Inputs, Outputs, Session, render, ui, reactive

#random data generator
data = reactive.Value(pd.DataFrame({"x_values": [], "y_values": []}))
def random_data() -> None:
    rand_m = np.random.uniform(-5,5)
    x_data = [np.random.randint(50, 200) for _ in range(100)]

    data.set(pd.DataFrame({
        "x_values": x_data,
        "y_values": [int(rand_m * xi + np.random.normal(2,15)) for xi in x_data] 
    }))
random_data()

#ui
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Linear Regression",
        ui.layout_sidebar(
            ui.sidebar(
                ui.output_data_frame("data_display"),
                ui.input_action_button("randomise_button", "Randomise Data"),
                #ui.input_action_button("regression_button", "Draw regression line"),
                ui.input_file("csvfile", "Upload your own CSV dataset", accept=[".csv"], multiple=False)
            ),
            
            ui.card(
                ui.card_header("Scatterplot"),
                ui.output_plot("scatter_plot"),  
                ui.input_slider("regression_slider", "Draw Regression Line", min=0, max=5, value=0, animate=True),  
                
            ),
            ui.card(
                ui.card_header("Equation of the line of best fit:"),
                ui.output_code("curr_equation_lr", placeholder=True),
                ui.output_code("final_equation_lr", placeholder=True),
            ),
            
        ),
    ),
    ui.nav_panel(
        "K-Means Clustering",
        "Hi",
    ),
    ui.nav_panel(
        "Decision Trees",
        "Hello",
    ),
)

#server
def server(input, output, session):

    show_regression = reactive.Value(False)

    @reactive.effect
    @reactive.event(input.regression_slider)
    def _() -> None:
        show_regression.set(True)

    curr_m = reactive.Value(0)    
    curr_b = reactive.Value(0)   

    final_m = reactive.Value(0)    
    final_b = reactive.Value(0)    
    @reactive.Calc
    def linear_regression():
        df = data().copy()
        df['raw_x_values'] = df['x_values'] #saving non-normalized x values
        df['x_values'] = (df['x_values']-df['x_values'].mean())/df['x_values'].std() #normalising x values

        m = 0
        b = 0
        L = 0.01
        epochs = 300 #iterations

        m_values = []
        b_values = []

        for i in range(epochs):
                m, b = gradient_descent(m, b, df, L)
                if i %50 ==0:
                    m_values.append(m)
                    b_values.append(b)
        final_m.set(m)         
        final_b.set(b)
        return m_values, b_values
        
    @render.data_frame
    def data_display():
        df = data()
        return render.DataGrid(df, width='200px', height='300px')

    @render.plot(alt="Scatterplot")
    def scatter_plot():
        df = data().copy()
        df['raw_x_values'] = df['x_values']
        fig, ax = plt.subplots()
        ax.scatter(df.x_values, df.y_values, color="black")
        ax.set_xlabel("X Values")
        ax.set_ylabel("Y Values")
        
        if show_regression():
            m_values, b_values = linear_regression()

            mu = df.x_values.mean()
            sigma = df.x_values.std()

            zipped_mb = list(zip(m_values, b_values))
            m,b = zipped_mb[input.regression_slider()]
            curr_m.set(m)         
            curr_b.set(b)

            m_raw = m / sigma
            b_raw = b - (m * mu / sigma)

            ax.plot(df.raw_x_values, m_raw * df.raw_x_values+ b_raw, color="red") #regression line


        return fig

    @reactive.effect
    @reactive.event(input.randomise_button)
    def _() -> None:
        random_data()
        show_regression.set(False)
        curr_m.set(0)
        curr_b.set(0)
        ui.update_slider("regression_slider", value=0)


    @render.code
    def curr_equation_lr():
       return f"Current equation: y = {round(curr_m.get(), 2)}x + {round(curr_b.get(), 2)}"
    
    @render.code
    def final_equation_lr():
       return f"Final equation: y = {round(final_m.get(), 2)}x + {round(final_b.get(), 2)}"

    
app = App(app_ui, server)