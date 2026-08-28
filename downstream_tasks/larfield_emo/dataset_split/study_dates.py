from datetime import date, timedelta

LARFIELD_DATES = {
    'i_01': [date(2023, 3, 6), date(2023, 4, 2)],
    'i_02': [date(2023, 4, 17), date(2023, 5, 14)],
    'i_03': [date(2023, 5, 22), date(2023, 6, 18)],
    'i_04': [date(2023, 6, 26), date(2023, 7, 23)],
    'i_05': [date(2023, 9, 11), date(2023, 10, 8)],
    'i_06': [date(2023, 10, 16), date(2023, 11, 12)],
    'i_07': [date(2023, 11, 20), date(2023, 12, 17)],
}


def get_study_dates(study_name: str) -> (date, date):
    if isinstance(study_name, int):  # or can be changed to int
        study_name = f"i_0{study_name}"

    if 'iteration' in study_name:
        study_name = f"i_0{study_name[-1]}"

    if study_name in LARFIELD_DATES.keys():
        return LARFIELD_DATES[study_name]
    else:
        print(f"{study_name} not in STUDY_DATES")
        return date.today() - timedelta(days=7), date.today()


def get_study_dates_range(study_date_start: date, study_date_end: date) -> [date]:
    return [study_date_start + timedelta(days=i) for i
            in range((study_date_end - study_date_start).days + 1)]
