from ai.input_manager import (
    ImageLoader,
    MetadataExtractor,
    InputValidator,
    InputMode,
    InputRequest
)


def test_input_manager():

    image_path = "datasets/raw/test_image.png"

    request = InputRequest(
        mode=InputMode.SINGLE,
        primary_image=image_path
    )

    print("\nSatQuery Input Manager Test")
    print("-" * 40)

    try:
        InputValidator().validate(request)

        image = ImageLoader().load(image_path)

        metadata = MetadataExtractor().extract(
            image_path,
            image
        )

        print("Input validation : PASS")
        print("Image loading    : PASS")
        print("Metadata         : PASS")

        print("\nMetadata")
        print("Filename :", metadata.filename)
        print("Width    :", metadata.width)
        print("Height   :", metadata.height)
        print("Channels :", metadata.channels)
        print("Data type:", metadata.dtype)
        print("Type     :", metadata.image_type.value)
        print("Format   :", metadata.format)

    except Exception as error:
        print("Test failed:")
        print(error)


if __name__ == "__main__":
    test_input_manager()