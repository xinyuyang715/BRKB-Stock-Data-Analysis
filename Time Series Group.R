############################################################
# PROJECT: Berkshire Hathaway Inc. Class B (BRK-B)
# TOPIC:   Monthly price forecasting (2019–2024)
# COURSE:  IS512 – Time Series Analysis
# FLOW:    Data -> TS -> ETS & ARIMA -> Evaluation -> Interpretation
############################################################

# --- 0. Packages -----------------------------------------------------------

# install.packages(c("tidyverse","lubridate","forecast","quantmod"))
library(tidyverse)   # data manipulation & plots
library(lubridate)   # date handling
library(forecast)    # ETS, ARIMA, accuracy()
library(quantmod)    # getSymbols from Yahoo Finance

# --- 1. Data collection: BRK-B from Yahoo Finance --------------------------
# We use Berkshire Hathaway Class B (BRK-B), a large, diversified holding
# company whose stock is widely followed in financial markets. 

options("getSymbols.warning4.0" = FALSE)
options("getSymbols.yahoo.warning" = FALSE)

from_date <- "2019-01-01"
to_date   <- "2024-12-31"

# For symbols with a dash like BRK-B we store the result in an object
# using auto.assign = FALSE to avoid naming problems.
brkb_xts <- getSymbols("BRK-B",
                       src         = "yahoo",
                       from        = from_date,
                       to          = to_date,
                       auto.assign = FALSE)
brkb_xts


brkb_df <- data.frame(Date = index(brkb_xts), coredata(brkb_xts))

write.csv(brkb_df, file = "BRKB_Stock_Data.csv", row.names = FALSE)

# --- 2. Data preprocessing & time series structuring -----------------------

# 2.1. Convert daily OHLCV xts to a data frame and keep Adjusted Close,
# which accounts for splits and dividends. 
brkb_daily_df <- brkb_xts %>%
  as.data.frame() %>%
  mutate(Date = as.Date(index(brkb_xts))) %>%
  arrange(Date) %>%
  select(Date, `BRK-B.Adjusted`)

# 2.2. Aggregate to MONTHLY data using the last trading day of each month. 
brkb_monthly_xts <- to.period(brkb_xts,
                              period  = "months",
                              indexAt = "lastof",
                              name    = "BRK-B")

brkb_monthly_df <- brkb_monthly_xts %>%
  as.data.frame() %>%
  mutate(Date = as.Date(index(brkb_monthly_xts))) %>%
  select(Date, `BRK-B.Adjusted`) %>%
  arrange(Date) %>%
  filter(Date >= as.Date("2019-01-01"),
         Date <= as.Date("2024-12-31"))

# 2.3. Build monthly time series object (frequency = 12). 
brkb_ts <- ts(brkb_monthly_df$`BRK-B.Adjusted`,
              start = c(2019, 1),
              frequency = 12)

# --- 3. Train–test split (80% / 20% in time) ------------------------------
# Project guidelines require model training and evaluation on a hold-out set. 

n_obs <- length(brkb_ts)
split_index <- floor(n_obs * 0.8)     # 80% train, 20% test

brkb_train <- window(brkb_ts,
                     end = time(brkb_ts)[split_index])
brkb_test  <- window(brkb_ts,
                     start = time(brkb_ts)[split_index + 1])

h <- length(brkb_test)                 # forecast horizon = test length

# --- 4. Exploratory visualization -----------------------------------------

dev.new()
autoplot(brkb_train) +
  ggtitle("BRK-B Adjusted Close (Monthly, Training Sample)") +
  xlab("Year") + ylab("Price (USD)")

# --- 5. Exponential Smoothing model (ETS) ----------------------------------
# ETS automatically chooses an appropriate exponential smoothing form

ets_fit_brkb <- ets(brkb_train)
ets_fit_brkb   # shows the selected ETS components

ets_fc_brkb <- forecast(ets_fit_brkb, h = h)

dev.new()
autoplot(ets_fc_brkb) +
  autolayer(brkb_test, series = "Actual (Test)") +
  ggtitle("ETS Forecast for BRK-B") +
  xlab("Year") + ylab("Price (USD)") +
  guides(colour = guide_legend(title = "Series"))

# --- 6. ARIMA model --------------------------------------------------------
# We use automatic Box–Jenkins selection with auto.arima(). 

arima_fit_brkb <- auto.arima(brkb_train,
                             seasonal      = TRUE,
                             stepwise      = TRUE,
                             approximation = FALSE)
arima_fit_brkb  # ARIMA(p,d,q)(P,D,Q)[12] with drift if selected

arima_fc_brkb <- forecast(arima_fit_brkb, h = h)

dev.new()
autoplot(arima_fc_brkb) +
  autolayer(brkb_test, series = "Actual (Test)") +
  ggtitle("ARIMA Forecast for BRK-B") +
  xlab("Year") + ylab("Price (USD)") +
  guides(colour = guide_legend(title = "Series"))

# --- 7. Forecast evaluation (RMSE, MAE, MAPE) ------------------------------
# The course requires model evaluation with accuracy metrics. 

acc_ets_brkb   <- accuracy(ets_fc_brkb,   brkb_test)
acc_arima_brkb <- accuracy(arima_fc_brkb, brkb_test)

acc_ets_brkb
acc_arima_brkb

ets_rmse  <- acc_ets_brkb["Test set", "RMSE"]
ets_mae   <- acc_ets_brkb["Test set", "MAE"]
ets_mape  <- acc_ets_brkb["Test set", "MAPE"]

arima_rmse <- acc_arima_brkb["Test set", "RMSE"]
arima_mae  <- acc_arima_brkb["Test set", "MAE"]
arima_mape <- acc_arima_brkb["Test set", "MAPE"]

model_compare_brkb <- tibble(
  Model = c("ETS (Exp. Smoothing)", "ARIMA"),
  RMSE  = c(ets_rmse,  arima_rmse),
  MAE   = c(ets_mae,   arima_mae),
  MAPE  = c(ets_mape,  arima_mape)
)

print(model_compare_brkb)

write_csv(model_compare_brkb, "model_comparison_BRKB_monthly_2019_2024.csv")

# --- 8. Residual diagnostics ----------------------------------------------

dev.new()
checkresiduals(ets_fit_brkb)

dev.new()
checkresiduals(arima_fit_brkb)

############################################################
# END OF SCRIPT
############################################################