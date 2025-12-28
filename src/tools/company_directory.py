import numpy as np
import pandas as pd
from langchain.tools import tool
from typing import cast

EMAILS = pd.read_csv("data/raw/email_addresses.csv", header=None, names=["email_address"])


@tool("company_directory.find_email_address", return_direct=False)
def find_email_address(name: str = "") -> str | np.ndarray[tuple[int], np.dtype[np.str_]]:
    """
    Finds the email address of an employee by their name.

    Parameters
    ----------
    name : str, optional
        Name of the person.

    Returns
    -------
    email_address : str
        Email addresses of the person.

    Examples
    --------
    >>> directory.find_email_address("John")
    "john.smith@example.com"
    """
    global EMAILS
    if name == "":
        return "Name not provided."
    name = name.lower()
    email_address_df = EMAILS[EMAILS["email_address"].str.contains(name)]
    email_series = cast(pd.Series, email_address_df["email_address"])
    result = email_series.values
    if len(result) == 1:
        return str(result[0])
    return cast(np.ndarray[tuple[int], np.dtype[np.str_]], result)
