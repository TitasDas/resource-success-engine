# Feature engineering with validated joins and cost computations via People.rate

import pandas as pd
import numpy as np
import logging
from src.etl import load_tables

logging.basicConfig(level=logging.INFO)

def safe_divide(a, b):
    return a / b.replace(0, np.nan)

def build_feature_view(return_shap_ready=False, override_db_path=None):
    tables = load_tables(override_db_path)
    projects = tables["Projects"]
    tasks = tables["Tasks"]
    allocated = tables["AllocatedTasks"]
    logged = tables["LoggedHours"]
    people = tables["People"]

    # Join to get project_id and rate
    alloc = allocated.merge(tasks[["id", "project_id"]], left_on="task_id", right_on="id", suffixes=("", "_task"))
    alloc = alloc.merge(people[["id", "rate"]], left_on="person_id", right_on="id", suffixes=("", "_person"))
    alloc["cost"] = alloc["hours"] * alloc["rate"]

    logs = logged.merge(tasks[["id", "project_id"]], left_on="task_id", right_on="id", suffixes=("", "_task"))
    logs = logs.merge(people[["id", "rate"]], left_on="person_id", right_on="id", suffixes=("", "_person"))
    logs["cost"] = logs["hours"] * logs["rate"]

    # Expected hours from weekly schedule
    weekday_cols = [
        "working_hours_monday", "working_hours_tuesday", "working_hours_wedday",
        "working_hours_thursday", "working_hours_friday"
    ]
    people["expected_daily_hours"] = people[weekday_cols].mean(axis=1)
    expected_map = people.set_index("id")["expected_daily_hours"].to_dict()
    logs["expected_hours"] = logs["person_id"].map(expected_map)

    alloc_group = alloc.groupby("project_id").agg(
        planned_hours_total=("hours", "sum"),
        allocated_task_count=("task_id", "nunique"),
        allocated_people_count=("person_id", "nunique"),
        avg_rate=("rate", "mean"),
        planned_cost=("cost", "sum")
    )

    log_group = logs.groupby("project_id").agg(
        logged_hours_total=("hours", "sum"),
        actual_cost=("cost", "sum"),
        overtime_hours=("hours", lambda x: ((x - logs.loc[x.index, "expected_hours"]).clip(lower=0)).sum())
    )

    pf = projects.set_index("id")[["budget", "status", "is_successful", "start_date", "end_date", "project_manager_id"]]         .join(alloc_group).join(log_group)

    pf["hours_logged_per_allocated"] = safe_divide(pf["logged_hours_total"], pf["planned_hours_total"])
    pf["cost_variance_pct"] = safe_divide(pf["actual_cost"] - pf["planned_cost"], pf["planned_cost"])
    pf["overtime_ratio"] = safe_divide(pf["overtime_hours"], pf["logged_hours_total"])

    pf["start_date"] = pd.to_datetime(pf["start_date"])
    pf["end_date"] = pd.to_datetime(pf["end_date"])
    pf["project_duration_days"] = (pf["end_date"] - pf["start_date"]).dt.days

    completed = projects[projects["status"] == "Completed"]
    pm_stats = completed.groupby("project_manager_id")["is_successful"].agg(["count", "mean"])
    pm_stats.columns = ["pm_completed_projects", "pm_success_rate"]
    pf = pf.join(pm_stats, on="project_manager_id")
    pf["pm_completed_projects"] = pf["pm_completed_projects"].fillna(0)
    pf["pm_success_rate"] = pf["pm_success_rate"].fillna(0)

    people["avg_weekday_hours"] = people[weekday_cols].mean(axis=1)
    people["is_full_time"] = (people["avg_weekday_hours"] >= 8).astype(int)
    pf["pm_is_full_time"] = pf["project_manager_id"].map(people.set_index("id")["is_full_time"]).fillna(0)

    pf["tasks_per_person"] = safe_divide(pf["allocated_task_count"], pf["allocated_people_count"])

    alloc_std = alloc.groupby(["project_id", "person_id"])["hours"].sum().groupby("project_id").std()
    pf = pf.join(alloc_std.rename("allocation_std_per_person"))

    alloc["date"] = pd.to_datetime(alloc["date"])
    alloc = alloc.merge(projects[["id", "start_date"]], left_on="project_id", right_on="id")
    alloc["start_date"] = pd.to_datetime(alloc["start_date"])
    alloc["allocation_lag_days"] = (alloc["date"] - alloc["start_date"]).dt.days
    lag = alloc.groupby("project_id")["allocation_lag_days"].mean()
    pf = pf.join(lag.rename("avg_allocation_lag_days"))

    spikiness = alloc.groupby(["project_id", "date"])["hours"].sum().groupby("project_id").std()
    pf = pf.join(spikiness.rename("allocation_spikiness"))

    task_ids_with_logs = set(logged["task_id"])
    total = tasks.groupby("project_id")["id"].count()
    drop = tasks.groupby("project_id")["id"].apply(lambda x: (~x.isin(task_ids_with_logs)).sum())
    pf["dropoff_rate"] = safe_divide(drop, total)

    active_days = alloc.groupby("project_id")["date"].nunique()
    pf["active_days"] = active_days
    pf["tasks_per_day"] = safe_divide(pf["allocated_task_count"], pf["active_days"])

    pf["log_vs_cost_ratio"] = safe_divide(pf["logged_hours_total"], pf["budget"])
    pf["pm_success_weighted"] = pf["pm_success_rate"] * pf["pm_completed_projects"]
    pf["avg_cost_per_person"] = safe_divide(pf["budget"], pf["allocated_people_count"])
    pf["cost_per_task"] = safe_divide(pf["budget"], pf["allocated_task_count"])
    pf["overtime_efficiency"] = safe_divide(pf["overtime_hours"], pf["overtime_ratio"])
    pf["is_overbudget"] = (pf["actual_cost"] > pf["budget"]).astype(int)
    pf["is_underlogged"] = (pf["hours_logged_per_allocated"] < 0.8).astype(int)
    pf["is_high_variance_pm"] = ((pf["pm_success_rate"] < 0.5) & (pf["pm_completed_projects"] > 5)).astype(int)
    pf["is_high_complexity"] = (pf["tasks_per_person"] > 3).astype(int)
    pf["duration_per_person"] = safe_divide(pf["project_duration_days"], pf["allocated_people_count"])
    pf["task_density"] = safe_divide(pf["allocated_task_count"], pf["project_duration_days"])

    pf = pf.fillna({col: 0 for col in pf.select_dtypes(exclude=["datetime64[ns]"]).columns})

    if return_shap_ready:
        X = pf.drop(columns=["is_successful", "status", "start_date", "end_date"])
        y = pf["is_successful"]
        return X, y, pf

    return pf

if __name__ == "__main__":
    df = build_feature_view()
    print(f"Built feature view with shape: {df.shape}")
    print(df.head())