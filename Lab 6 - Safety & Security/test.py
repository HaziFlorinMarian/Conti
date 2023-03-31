if __name__ == '__main__':
    import rsa_library

    public_key, private_key = rsa_library.generate_keypair(277, 239)
    print("Public key: ", public_key)
    print("Private key: ", private_key)
    hex_number = str(rsa_library.NOT_low) + str(rsa_library.ON_low[2:])
    print("Hex number: ", hex_number)
    encrypted_msg = rsa_library.encrypt(public_key, hex_number)
    print("Encrypted message: ", encrypted_msg)
    decrypted_msg = rsa_library.decrypt(private_key, encrypted_msg)
    print("Decrypted message: ", decrypted_msg)
    print(f"Low part of the hex number is LOW: {rsa_library.low_check(decrypted_msg)}")
    print(f"High part of the hex number is ~LOW: {rsa_library.number_check(decrypted_msg)}")