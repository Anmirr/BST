from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

def sign():
    def sign_file(file_path, private_key_path, signature_path):
        # Wczytanie pliku tekstowego
        with open(file_path, 'rb') as file:
            file_data = file.read()

        # Wczytanie klucza prywatnego w formacie PEM
        with open(private_key_path, 'rb') as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
                backend=default_backend()
            )

        # Podpisanie danych
        signature = private_key.sign(
            file_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        # Zapis podpisu do pliku
        with open(signature_path, 'wb') as signature_file:
            signature_file.write(signature)

    # Przykładowe użycie
    file_to_sign = 'file.txt'
    private_key_file = 'private_key.pem'
    signature_file = 'signature.bin'

    sign_file(file_to_sign, private_key_file, signature_file)