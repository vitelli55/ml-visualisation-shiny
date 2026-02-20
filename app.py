import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, render, ui, reactive

#local files
from models import LinearRegression, KMeansClustering
from random_data import linear_regression_df, kmeans_df

#random data generator
lrdata = reactive.Value(linear_regression_df())
kdata = reactive.Value(kmeans_df())

#ui
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Linear Regression",
        ui.layout_sidebar(
            ui.sidebar(
                ui.output_data_frame("lr_data_display"),
                ui.input_action_button("randomise_lr_button", "Randomise Data"),
                #ui.input_action_button("regression_button", "Draw regression line"),
                #ui.input_file("csvfile", "Upload your own CSV dataset", accept=[".csv"], multiple=False)
            ),
            
            ui.layout_columns(
                ui.card(
                    ui.card_header("Scatterplot"),
                    ui.output_plot("lr_scatter_plot", width=600),  
                    ui.input_slider("regression_slider", "Draw Regression Line", min=0, max=5, value=0, animate=True),  
                    
                ),
                ui.card(
                    ui.card_header("Equation of the line of best fit:"),
                    ui.output_code("lr_curr_equation", placeholder=True),
                    ui.output_code("lr_final_equation", placeholder=True),
                ),
            ),
        ),
    ),
    ui.nav_panel(
        "K-Means Clustering",
        ui.layout_sidebar(
            ui.sidebar(
                ui.output_data_frame("k_data_display"),
                ui.input_action_button("randomise_k_button", "Randomise Data"),
            ),
            
            ui.layout_columns(
                ui.card(
                    ui.card_header("Scatterplot"),
                    ui.output_plot("k_scatter_plot", width=600),  
                    ui.input_slider("k_button", "Make Clusters", min=0, max=10, value=0, animate=True),  
                    
                ),
                ui.card(
                    ui.card_header("Behind the scenes:"),
                    ui.output_code("centroids_positions", placeholder=True),
                )
            ),
        ),
    ),
    ui.nav_panel(
        "Decision Trees",
        "Work in progress...",
    ),
)

#server
def server(input, output, session):

    #LINEAR REGRESSION

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
    def fitted_linear_regression():
        df = lrdata().copy()
        df['raw_x_values'] = df['x_values'] #saving non-normalized x values
        df['x_values'] = (df['x_values']-df['x_values'].mean())/df['x_values'].std() #normalising x values

        model = LinearRegression()
        model.fit(df, epochs=300)
        return model
        
    @render.data_frame
    def lr_data_display():
        df = lrdata()
        return render.DataGrid(df, width='200px', height='300px')

    @render.plot(alt="Scatterplot")
    def lr_scatter_plot():
        df = lrdata().copy()
        df['raw_x_values'] = df['x_values']
        fig, ax = plt.subplots()
        ax.scatter(df.x_values, df.y_values, color="black")
        ax.set_xlabel("X Values")
        ax.set_ylabel("Y Values")
        
        if show_regression():
            linear_reg = fitted_linear_regression()

            mu = df.x_values.mean()
            sigma = df.x_values.std()

            zipped_mb = list(zip(linear_reg.m_values, linear_reg.b_values))
            m,b = zipped_mb[input.regression_slider()]

            curr_m.set(m)         
            curr_b.set(b)
            final_m.set(linear_reg.m_values[len(linear_reg.m_values)-1])
            final_b.set(linear_reg.b_values[len(linear_reg.b_values)-1])

            m_raw = m / sigma
            b_raw = b - (m * mu / sigma)

            ax.plot(df.raw_x_values, m_raw * df.raw_x_values+ b_raw, color="red") #regression line


        return fig

    @reactive.effect
    @reactive.event(input.randomise_lr_button)
    def _() -> None:
        lrdata.set(linear_regression_df())
        show_regression.set(False)
        curr_m.set(0)
        curr_b.set(0)
        ui.update_slider("regression_slider", value=0)


    @render.code
    def lr_curr_equation():
       return f"Current equation: y = {round(curr_m.get(), 2)}x + {round(curr_b.get(), 2)}"
    
    @render.code
    def lr_final_equation():
       return f"Final equation: y = {round(final_m.get(), 2)}x + {round(final_b.get(), 2)}"
    

    #KMEANS CLUSTERING

    show_clustering = reactive.Value(False)
    centroid_reactive = reactive.Value(None)
    #labels_reactive = reactive.Value(None)


    @reactive.effect
    @reactive.event(input.k_button)
    def _() -> None:
        show_clustering.set(True)

    @render.data_frame
    def k_data_display():
        df = kdata()
        return render.DataGrid(df, width='200px', height='300px')
    
    @reactive.calc #fitting model outside the plot function so the model only recomputes when the kdata() changes
    def fitted_kmeans():
        df = kdata().copy()
        nparr = df.to_numpy()

        kmeans = KMeansClustering(k=3)
        kmeans.fit(nparr)

        return kmeans

    @render.plot(alt="Scatterplot")
    def k_scatter_plot():
        df = kdata().copy()

        fig, ax = plt.subplots()
        ax.scatter(df.x_values, df.y_values, color="black")
        ax.set_xlabel("X Values")
        ax.set_ylabel("Y Values")

        if show_clustering():
            kmeans = fitted_kmeans()

            ui.update_slider("k_button", max=len(kmeans.history)-1)

            i = min(input.k_button(), len(kmeans.history)-1)
            centroids, labels = kmeans.history[i]
            centroid_reactive.set(centroids)
            #labels_reactive.set(labels)

            ax.scatter(df.x_values, df.y_values, c=labels)
            ax.scatter(centroids[:,0], centroids[:,1], marker="*", s=200) 

        return fig
    
    @reactive.effect
    @reactive.event(input.randomise_k_button)
    def _() -> None:
        kdata.set(kmeans_df())
        show_clustering.set(False)
        ui.update_slider("k_button", value=0)

    @render.code
    def centroids_positions():
        centroids = centroid_reactive.get()
        return f"1st centroids position:{centroids[0]} \n2nd centroid position:{centroids[1]} \n3rd centroid position:{centroids[2]}"



app = App(app_ui, server)