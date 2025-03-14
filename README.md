# Risk management for event companies
`TODO:` Consider new title: "Risk management using Deep Learning"
This repository contains a predictive model to estimate daily delay times at each station.
This model will help event-planning companies optimize event dates and shuttle bus schedules by providing accurate delay forecasts.
The goal is to enhance scheduling efficiency, minimize transportation disruptions, and improve attendee experience.

## Team members

This project was developed collaboratively by the following team members:  
-   Oleksii Morozenko [![LinkedIn](https://icons.iconarchive.com/icons/limav/flat-gradient-social/32/Linkedin-icon.png)](https://www.linkedin.com/in/oleksii-morozenko) [![GitHub](https://icons.iconarchive.com/icons/pictogrammers/material/32/github-icon.png)](https://github.com/oleksiimorozenko)
    
-   Colin (Lin) Song [![LinkedIn](https://icons.iconarchive.com/icons/limav/flat-gradient-social/32/Linkedin-icon.png)](https://www.linkedin.com/in/colin（lin）-song-msc-8a5b8611b) [![GitHub](https://icons.iconarchive.com/icons/pictogrammers/material/32/github-icon.png)](https://github.com/colsong)

    
-   Ian Lilley [![LinkedIn](https://icons.iconarchive.com/icons/limav/flat-gradient-social/32/Linkedin-icon.png)](https://www.linkedin.com/in/ian-lilley) [![GitHub](https://icons.iconarchive.com/icons/pictogrammers/material/32/github-icon.png)](https://github.com/ian-lilley)

    
-   Urooj Iftikhar [![LinkedIn](https://icons.iconarchive.com/icons/limav/flat-gradient-social/32/Linkedin-icon.png)](https://www.linkedin.com/in/urooj-fatima-iftikhar) [![GitHub](https://icons.iconarchive.com/icons/pictogrammers/material/32/github-icon.png)](https://github.com/Urooj1607)
    
-   Mara Di Loreto  [![LinkedIn](https://icons.iconarchive.com/icons/limav/flat-gradient-social/32/Linkedin-icon.png)](https://www.linkedin.com/in/maradiloreto) [![GitHub](https://icons.iconarchive.com/icons/pictogrammers/material/32/github-icon.png)](https://github.com/maradiloreto)
    
-   `TODO`: Brijesh

## <a id="project-overview"></a>Project Overview
- [Requirements](#requirements)
- [Target audience](#target-audience)
- [Understanding raw data](#understanding-raw-data)
- [Exploratory data analysis](#exploratory-data-analysis)
- [Data cleaning and handling missing values](#data-cleaning-and-handling-missing-values)
- [Data analysis](#data-analysis)
- [Conclusion](#conclusion)

### <a id="requirements"></a>Requirements
The project uses the following Python libraries:
- `NumPy`: Fast matrix operations
- `Pandas`: Dataset analysis
- `matplotlib`: Creating graphs and plots
- `seaborn`: Enhance matplotlib plots style
- `sklearn`: Linear regression analysis    
- `TODO: <review and rework this list as we end up with the exact list>` 

[< back to Overview](#project-overview)

### <a id="target-audience"></a>Target audience
`TODO: <determine target audience(s)>`

[< back to Overview](#project-overview)

### <a id="understanding-raw-data"></a>Understanding raw data
The [TTC Subway Delay Data](https://open.toronto.ca/dataset/ttc-subway-delay-data) was used for the project. It was decided to choose the period from 2022 to 2024 as the one with most consistent data points. 
It contains around 70 thousand individual data points, each representing daily subway delays over `TODO: <clarify the exact number>` subway stations.

Key features used for prediction include:
- Highest Delay time
- Lowest Delay time
- Stations with highest delays
- Stations with lowest delays

The dataset includes adjustments for `TODO: <add numbers>` splits, recorded in the `TODO: <add filename>` file. This ensures that the historical data is comparable to the latest data shared by the Toronto Open Data Portal, making it easier to analyze the performance over time.

[< back to Overview](#project-overview)

### <a id="exploratory-data-analysis"></a>Exploratory data analysis
- **Understanding the dataset:** We analyze the dataset by examining the number of rows and columns, identifying variable types (numerical, categorical, datetime, etc.), and understanding its source or context.
- **Checking for missing data:** We identify missing values using methods like `df.isnull().sum()`, and apply appropriate handling techniques such as removal `TODO: <discuss and clarify that>`

[< back to Overview](#project-overview)

### <a id="data-cleaning-and-handling-missing-values"></a>Data cleaning and handling missing values
Identifying top `TODO: <clarify number>` stations
Identifying factors we can predict (Controllable or Measurable). 
`TODO: <discuss how we group data by problem codes>`

[< back to Overview](#project-overview)

### <a id="data-analysis"></a>Data analysis
Techniques used for data analysis and the model learning:
- Linear Regression
- Clustering
- Forecasting

[< back to Overview](#project-overview)

### Model Selection and Training
- `TODO: <add details about the model that was used>`
- `TODO: <add details about the model was trained to prevent overfitting>`
- `TODO: describe how the training process was conducted (e.g. with <epoch_count> of epochs with a batch size of <batch_size>)`

[< back to Overview](#project-overview)

### <a id="conclusion"></a>Conclusion
`TODO: <add reflections>`

[< back to Overview](#project-overview)
