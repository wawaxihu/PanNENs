
library(shiny)
library(caret)
library(gbm)
library(ggplot2)
library(dplyr)

GBM_model <- readRDS("GBM_model.rds")

year_choices      <- c("2000-2007" = 1, "2008-2015" = 2, "2016-2023" = 3)
race_choices      <- c("Black" = 1, "Other" = 2, "White" = 3)
site_choices      <- c("Body" = 1, "Head" = 2, "Other" = 3, "Tail" = 4)
grade_choices     <- c("Well differentiated (G1)" = 1,
                       "Moderately differentiated (G2)" = 2,
                       "Poorly differentiated (G3)" = 3)
histology_choices <- c("Atypical carcinoid tumor" = 1,
                       "Carcinoid tumor" = 2,
                       "Neuroendocrine carcinoma" = 3,
                       "Other" = 4)
marital_choices   <- c("Married" = 1, "Unmarried" = 2)
sex_choices       <- c("Female" = 1, "Male" = 2)
income_choices    <- c("Low (<$80,000)" = 1,
                       "Middle ($80,000–$100,000)" = 2,
                       "High (>$100,000)" = 3)
residence_choices <- c("Rural" = 1, "Urban" = 2)


ui <- fluidPage(
  
  tags$head(
    tags$style(HTML("
      body { background-color: #f4f6f9; font-family: Arial, sans-serif; }
      .title-panel {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        color: white; padding: 20px 30px; border-radius: 8px;
        margin-bottom: 20px;
      }
      .title-panel h2 { margin: 0; font-size: 20px; font-weight: bold; }
      .title-panel p  { margin: 6px 0 0; font-size: 13px; opacity: 0.85; }
      .sidebar-card {
        background: white; border-radius: 8px;
        padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      }
      .result-card {
        background: white; border-radius: 8px;
        padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 16px;
      }
      .prob-display {
        text-align: center; padding: 20px;
        border-radius: 8px; margin: 10px 0;
      }
      .prob-number { font-size: 52px; font-weight: bold; }
      .risk-badge {
        display: inline-block; padding: 6px 20px;
        border-radius: 20px; font-size: 15px;
        font-weight: bold; color: white; margin-top: 8px;
      }
      .btn-predict {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        color: white; border: none; border-radius: 6px;
        width: 100%; padding: 12px; font-size: 15px;
        font-weight: bold; margin-top: 10px; cursor: pointer;
      }
      .btn-predict:hover { opacity: 0.9; color: white; }
      .section-title {
        font-size: 12px; font-weight: bold; color: #1a5276;
        text-transform: uppercase; letter-spacing: 0.5px;
        border-bottom: 2px solid #2e86c1;
        padding-bottom: 4px; margin: 14px 0 10px;
      }
      label { font-weight: 600; font-size: 13px; color: #34495e; }
      .disclaimer {
        font-size: 11px; color: #999; text-align: center;
        margin-top: 12px; line-height: 1.5;
      }
    "))
  ),
  
  div(class = "title-panel",
      tags$h2("Lymph Node Metastasis Risk Prediction in PanNETs"),
      tags$p("Pancreatic Neuroendocrine Tumors · GBM Machine Learning Model · SEER 2000–2023")
  ),
  
  fluidRow(
    
    column(4,
           div(class = "sidebar-card",
               h5("Patient Parameters",
                  style = "font-weight:bold; color:#1a5276; margin-bottom:4px;"),
               
               div(class = "section-title", "Tumor Characteristics"),
               selectInput("Grade", "Tumor Grade",
                           choices = grade_choices, selected = 1),
               selectInput("Site", "Tumor Location",
                           choices = site_choices, selected = 1),
               numericInput("Size", "Tumor Size (mm)",
                            min = 0, max = 500, value = 20, step = 1),
               selectInput("Histology", "Histological Type",
                           choices = histology_choices, selected = 2),
               
               div(class = "section-title", "Patient Demographics"),
               sliderInput("Age", "Age (years)",
                           min = 18, max = 95, value = 55, step = 1),
               selectInput("Sex", "Sex",
                           choices = sex_choices, selected = 1),
               selectInput("Race", "Race",
                           choices = race_choices, selected = 3),
               selectInput("Marital", "Marital Status",
                           choices = marital_choices, selected = 1),
               
               div(class = "section-title", "Socioeconomic Factors"),
               selectInput("Income", "Household Income",
                           choices = income_choices, selected = 2),
               selectInput("Residence", "Residence",
                           choices = residence_choices, selected = 2),
               selectInput("Year", "Diagnosis Year",
                           choices = year_choices, selected = 2),
               
               actionButton("goButton", "Predict LNM Risk",
                            class = "btn-predict"),
               
               div(class = "disclaimer",
                   "For research reference only.",
                   br(), "Not a substitute for clinical judgment.")
           )
    ),
    
    column(8,
           div(class = "result-card",
               h5("Prediction Result",
                  style = "font-weight:bold; color:#1a5276; margin-bottom:12px;"),
               uiOutput("result_summary")
           ),
           div(class = "result-card",
               h5("Probability Distribution",
                  style = "font-weight:bold; color:#1a5276; margin-bottom:4px;"),
               plotOutput("piediagram", height = "320px")
           ),
           div(class = "result-card",
               h5("Risk Interpretation",
                  style = "font-weight:bold; color:#1a5276; margin-bottom:8px;"),
               uiOutput("risk_interpretation")
           )
    )
  )
)

server <- function(input, output, session) {
  
  pred_result <- eventReactive(input$goButton, {
    
    new_data <- data.frame(
      Year      = as.integer(input$Year),
      Race      = as.integer(input$Race),
      Site      = as.integer(input$Site),
      Grade     = as.integer(input$Grade),
      Histology = as.integer(input$Histology),
      Marital   = as.integer(input$Marital),
      Sex       = as.integer(input$Sex),
      Income    = as.integer(input$Income),
      Residence = as.integer(input$Residence),
      Age       = as.numeric(input$Age),
      Size      = as.numeric(input$Size),
      stringsAsFactors = FALSE
    )
    
    prob  <- predict(GBM_model, newdata = new_data, type = "prob")
    p_yes <- as.numeric(prob[1, "Yes"])
    
    cutoff <- 0.253
    
    risk_level <- case_when(
      p_yes >= cutoff ~ "High Risk",
      TRUE            ~ "Low Risk"
    )
    risk_color <- case_when(
      p_yes >= cutoff ~ "#c0392b",
      TRUE            ~ "#27ae60"
    )
    risk_bg <- case_when(
      p_yes >= cutoff ~ "#fdedec",
      TRUE            ~ "#eafaf1"
    )
    
    list(
      p_yes      = p_yes,
      p_no       = 1 - p_yes,
      risk_level = risk_level,
      risk_color = risk_color,
      risk_bg    = risk_bg,
      cutoff     = cutoff
    )
  })
  
  output$risk_interpretation <- renderUI({
    
    if (input$goButton == 0) return(NULL)
    
    r <- pred_result()
    
    interpretation <- ifelse(
      r$p_yes >= r$cutoff,
      "High risk of lymph node metastasis. Systematic regional
       lymphadenectomy is recommended during pancreatectomy.
       Enhanced postoperative imaging surveillance and multidisciplinary
       team discussion are advised.",
      "Low risk of lymph node metastasis. Limited lymphadenectomy
       may be appropriate. Standard postoperative follow-up
       is recommended."
    )
    
    div(
      style = paste0("border-left: 4px solid ", r$risk_color,
                     "; padding: 10px 16px; background:", r$risk_bg, ";",
                     "border-radius: 4px; font-size: 13px; color: #2c3e50;"),
      interpretation
    )
  })

  output$piediagram <- renderPlot({
    
    if (input$goButton == 0) {
      ggplot() +
        annotate("text", x = 0.5, y = 0.5,
                 label = "Click 'Predict LNM Risk' to see results",
                 size = 5, color = "grey60") +
        theme_void()
    } else {
      
      r <- pred_result()
      
      pie_df <- data.frame(
        Category    = c("LNM Positive", "LNM Negative"),
        Probability = c(r$p_yes, r$p_no),
        Color       = c(r$risk_color, "#85c1e9")
      )
      
      ggplot(pie_df, aes(x = "", y = Probability, fill = Category)) +
        geom_bar(stat = "identity", width = 1,
                 color = "white", linewidth = 1) +
        coord_polar(theta = "y", start = 0) +
        scale_fill_manual(
          values = setNames(pie_df$Color, pie_df$Category)
        ) +
        geom_text(
          aes(label = paste0(round(Probability * 100, 1), "%")),
          position = position_stack(vjust = 0.5),
          size = 6, color = "white", fontface = "bold"
        ) +
        labs(
          title = paste0("Risk Level: ", r$risk_level,
                         "  |  P(LNM) = ",
                         round(r$p_yes * 100, 1), "%"),
          fill = NULL
        ) +
        theme_void(base_size = 13) +
        theme(
          plot.title      = element_text(hjust = 0.5, face = "bold",
                                         size = 13,
                                         margin = margin(b = 12)),
          legend.position = "bottom",
          legend.text     = element_text(size = 11)
        )
    }
  })
  
  output$risk_interpretation <- renderUI({
    
    if (input$goButton == 0) return(NULL)
    
    r <- pred_result()
    
    interpretation <- case_when(
      r$p_yes >= 0.60 ~
        "High risk of lymph node metastasis. Consider systematic regional
         lymphadenectomy during pancreatectomy and close postoperative
         surveillance with enhanced imaging follow-up.",
      r$p_yes >= r$cutoff ~
        "Intermediate risk of lymph node metastasis. Individualized
         surgical planning recommended; multidisciplinary team discussion
         advised for lymph node dissection extent.",
      TRUE ~
        "Low risk of lymph node metastasis. Limited lymphadenectomy may
         be appropriate; standard postoperative follow-up recommended."
    )
    
    div(
      style = paste0("border-left: 4px solid ", r$risk_color,
                     "; padding: 10px 16px; background:", r$risk_bg, ";",
                     "border-radius: 4px; font-size: 13px; color: #2c3e50;"),
      interpretation
    )
  })
}

shinyApp(ui = ui, server = server)