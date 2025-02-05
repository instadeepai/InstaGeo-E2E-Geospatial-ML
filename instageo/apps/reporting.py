import folium
from email.mime.multipart import MIMEMultipart  # Import for multipart messages
import io
from PIL import Image


def generate_high_density_report(fig: folium.Map):
    """Generates a report of locations with density above the given threshold."""
    img_data = fig._to_png(5)
    img = Image.open(io.BytesIO(img_data))
    map_image_path = "temp_map.png"
    img.save(map_image_path)
    return map_image_path


def send_email(to_email, subject, img_path):
    """Sends an email. Replace with your preferred email sending method."""
    try:
        import smtplib
        from email.mime.image import MIMEImage

        from_mail = "locustbusters@gmail.com"
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = from_mail
        msg["To"] = to_email

        with open(img_path, "rb") as img_file:
            img = MIMEImage(img_file.read(), _subtype="png")
            img.add_header("Content-ID", f"<{img_path}>")
            msg.attach(img)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_mail, "rvabulmpjdtuwvza")
            server.send_message(msg)

        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
