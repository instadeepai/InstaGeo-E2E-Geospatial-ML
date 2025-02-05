import folium
from email.mime.multipart import MIMEMultipart
import io
import pandas as pd
from PIL import Image


def generate_high_density_report(
    fig: folium.Map,
    all_clusters: pd.DataFrame,
    year: int,
    month: int,
):
    """Generates a report of locations with density above the given threshold."""
    img_data = fig._to_png(5)
    img = Image.open(io.BytesIO(img_data))
    map_image_path = "temp_map.png"
    img.save(map_image_path)

    report_text = generate_report_text(all_clusters, year, month)
    return map_image_path, report_text


def generate_report_text(all_clusters: pd.DataFrame, year: int, month: int) -> str:
    """Generates the text content for the report (simplified)."""

    report_text = f"Locust risk warning for {month}/{year}:\n\n"

    high_risk_clusters = all_clusters[all_clusters["risk"].isin(["high", "very high"])]

    if not high_risk_clusters.empty:
        for (
            index,
            row,
        ) in high_risk_clusters.iterrows():
            mean_lat = row.geometry.centroid.y
            mean_lon = row.geometry.centroid.x
            report_text += f"- {row['risk'].title()} risk around this area: ({mean_lat:.4f}, {mean_lon:.4f})\n"
    else:
        report_text += "- No high or very high risk areas detected.\n"

    return report_text


def send_email(to_email, subject, img_path, report_text, year: int, month: int):
    """Sends an email. Replace with your preferred email sending method."""
    try:
        import smtplib
        from email.mime.image import MIMEImage
        from email.mime.text import MIMEText

        from_mail = "locustbusters@gmail.com"
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = from_mail
        msg["To"] = to_email

        body = report_text
        text_part = MIMEText(body, "plain")
        msg.attach(text_part)

        with open(img_path, "rb") as img_file:
            img = MIMEImage(img_file.read(), _subtype="png")
            img_disposition = (
                f'attachment; filename="locust_warning_{year}_{month}.png"'
            )
            img.add_header("Content-Disposition", img_disposition)

            msg.attach(img)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_mail, "rvabulmpjdtuwvza")
            server.send_message(msg)

        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
