import pandas as pd

from src.tools import email

# Sample data for emails
test_emails = [
    {
        "email_id": "12345678",
        "inbox/outbox": "inbox",
        "sender/recipient": "jane@example.com",
        "subject": "Project Update",
        "sent_datetime": "2024-01-10 09:30:00",
        "body": "Please find the project update attached.",
    },
    {
        "email_id": "12345679",
        "inbox/outbox": "inbox",
        "sender/recipient": "mark@example.com",
        "subject": "Meeting Request",
        "sent_datetime": "2024-01-11 10:15:00",
        "body": "Can we schedule a meeting for next week?",
    },
]


def test_get_email_information_by_id():
    """
    Tests get_email_information_by_id.
    """
    email.EMAILS = pd.DataFrame(test_emails)
    assert get_function_from_tool(email.get_email_information_by_id)("12345678", "subject") == {
        "subject": "Project Update"
    }
    email.reset_state()


def test_get_email_information_missing_arguments():
    """
    Tests get_email_information_by_id with no ID and no field.
    """
    email.EMAILS = pd.DataFrame(test_emails)
    assert get_function_from_tool(email.get_email_information_by_id)() == "Email ID not provided."
    assert get_function_from_tool(email.get_email_information_by_id)("12345678") == "Field not provided."
    email.reset_state()


def test_get_email_information_by_id_field_not_found():
    """
    Tests get_email_information_by_id with field not found.
    """
    email.EMAILS = pd.DataFrame(test_emails)
    result = get_function_from_tool(email.get_email_information_by_id)("12345678", "field_does_not_exist")
    assert result == "Field not found."
    email.reset_state()


def test_search_emails():
    """
    Tests search_emails.
    """
    email.EMAILS = pd.DataFrame(test_emails)
    assert get_function_from_tool(email.search_emails)("Meeting Request")[0] == {
        "email_id": "12345679",
        "inbox/outbox": "inbox",
        "sender/recipient": "mark@example.com",
        "subject": "Meeting Request",
        "sent_datetime": "2024-01-11 10:15:00",
        "body": "Can we schedule a meeting for next week?",
    }
    email.reset_state()


def test_search_emails_none_found():
    """
    Tests search_emails with no emails found.
    """
    email.EMAILS = pd.DataFrame(test_emails)
    assert get_function_from_tool(email.search_emails)("email_does_not_exist") == "No emails found."
    email.reset_state()


def test_search_emails_multiple_fields_at_once():
    """
    Tests search_emails with multiple fields at once, for example, searching for both a name and an email subject at the same time.
    """
    email.EMAILS = pd.DataFrame(test_emails)
    assert get_function_from_tool(email.search_emails)("Mark Meeting Request")[0] == {
        "email_id": "12345679",
        "inbox/outbox": "inbox",
        "sender/recipient": "mark@example.com",
        "subject": "Meeting Request",
        "sent_datetime": "2024-01-11 10:15:00",
        "body": "Can we schedule a meeting for next week?",
    }


def test_search_emails_no_results():
    """
    Tests search_emails with no results.
    """
    assert get_function_from_tool(email.search_emails)("email_does_not_exist") == "No emails found."


def test_send_email():
    """
    Tests send_email.
    """
    assert (
        get_function_from_tool(email.send_email)("jane@example.com", "Reminder", "Meeting at 10am")
        == "Email sent successfully."
    )
    # check that the email was added to the outbox
    assert email.EMAILS["inbox/outbox"].values[-1] == "outbox"
    assert email.EMAILS["sender/recipient"].values[-1] == "jane@example.com"
    assert email.EMAILS["subject"].values[-1] == "Reminder"
    assert email.EMAILS["body"].values[-1] == "Meeting at 10am"
    email.reset_state()


def test_send_email_missing_args():
    """
    Tests send_email with missing arguments.
    """
    assert get_function_from_tool(email.send_email)() == "Recipient, subject, or body not provided."
    assert get_function_from_tool(email.send_email)("jane@example.com") == "Recipient, subject, or body not provided."


def test_delete_email():
    """
    Tests delete_email.
    """
    email.EMAILS = pd.DataFrame(test_emails)
    assert get_function_from_tool(email.delete_email)("12345678") == "Email deleted successfully."
    assert "12345678" not in email.EMAILS["email_id"].values
    email.reset_state()


def test_delete_email_no_id_provided():
    """
    Tests delete_email with no email_id provided.
    """
    assert get_function_from_tool(email.delete_email)() == "Email ID not provided."


def test_delete_email_not_found():
    """
    Tests delete_email with an email_id that does not exist.
    """
    email.EMAILS = pd.DataFrame(test_emails)
    assert get_function_from_tool(email.delete_email)("00000000") == "Email not found."
    email.reset_state()


def test_forward_email():
    """
    Tests forward_email.
    """
    email.EMAILS = pd.DataFrame(test_emails)
    assert (
        get_function_from_tool(email.forward_email)("12345679", "example@email.com") == "Email forwarded successfully."
    )
    # Check that the email was added to the outbox
    assert email.EMAILS["inbox/outbox"].values[-1] == "outbox"
    assert email.EMAILS["sender/recipient"].values[-1] == "example@email.com"
    assert email.EMAILS["subject"].values[-1] == "FW: Meeting Request"
    assert email.EMAILS["body"].values[-1] == "Can we schedule a meeting for next week?"
    email.reset_state()


def test_forward_email_missing_args():
    """
    Tests forward_email with missing arguments.
    """
    assert get_function_from_tool(email.forward_email)() == "Email ID or recipient not provided."
    assert get_function_from_tool(email.forward_email)("12345679") == "Email ID or recipient not provided."
    assert get_function_from_tool(email.forward_email)(recipient="example@email.com") == (
        "Email ID or recipient not provided."
    )


def test_reply_email():
    """
    Tests reply_email.
    """
    email.EMAILS = pd.DataFrame(test_emails)
    assert (
        get_function_from_tool(email.reply_email)("12345678", "Thank you for the update.")
        == "Email replied successfully."
    )
    # Check that the email was added to the outbox
    assert email.EMAILS["inbox/outbox"].values[-1] == "outbox"
    assert email.EMAILS["sender/recipient"].values[-1] == "jane@example.com"
    assert email.EMAILS["subject"].values[-1] == "Project Update"
    assert email.EMAILS["body"].values[-1] == "Thank you for the update."
    email.reset_state()


def test_reply_email_missing_args():
    """
    Tests reply_email with missing arguments.
    """
    assert get_function_from_tool(email.reply_email)() == "Email ID or body not provided."
    assert get_function_from_tool(email.reply_email)("12345678") == "Email ID or body not provided."
    assert (
        get_function_from_tool(email.reply_email)(body="Thank you for the update.") == "Email ID or body not provided."
    )
