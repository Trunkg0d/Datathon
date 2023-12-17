import streamlit as st
from PIL import Image
from main import pose_transfer, layering, tucking_in_out

UPLOADED_FILE = "data/test"
OUTPUT_FILE = "output"
uploaded_file = None
target_file = None


# 'top':5, # dress is also considered as top.
# 'bottom':1,
# 'hair':2,
# 'jacket':3
link_chatbot = "https://9ede-115-73-213-165.ngrok-free.app/?fbclid=IwAR3XBe74e6J9WGtz8nnEMnMnYuSS6PF2QIrViasuutnjEz8PfCPWrDycrLw"
with st.sidebar:
    chatbot = st.link_button(label="Chatbot", url=link_chatbot)


def main():
    global uploaded_file, target_file  # Make variables global

    st.title("Virtual Fitting Room")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    # Initialize session state variables
    if 'filename' not in st.session_state:
        st.session_state["filename"] = ""

    if 'target_filename' not in st.session_state:
        st.session_state["target_filename"] = ""

    if uploaded_file is not None:
        # Display the uploaded image
        uploaded_image = st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Processing options
        option = st.radio("Select Processing Option", ["Layering", "Multi-Layering", "Tucking In/Out", "Pose Transfer"])

        # Save the uploaded image based on the selected option
        # save_button = st.button("Choose")
        # if save_button:
        #     uploaded_image.empty()  # Empty the space for the uploaded image
        #     process_features(uploaded_file, option)
        process_features(uploaded_file, option)


def process_features(uploaded_file, option):
    global uploaded_image  # Make variable global

    # Extract filename from the uploaded file
    st.session_state.filename = uploaded_file.name

    # Open and process the image based on the selected option
    image = Image.open(uploaded_file)
    image.save(UPLOADED_FILE + "/" + f"{st.session_state.filename}")

    # Perform processing based on the selected option
    if option == "Layering":
        # Add layering processing code here
        layering_parameters()
    elif option == "Multi-Layering":
        # Add layering processing code here
        multi_layering_parameters()
    elif option == "Tucking In/Out":
        # Add tucking in/out processing code here
        tuckinout_parameter()
    elif option == "Pose Transfer":
        pose_transfer_parameters()


# POSE TRANSFER
def pose_transfer_parameters():
    global target_image  # Make variable global

    st.subheader("Pose Transfer Parameters")

    target_file = st.file_uploader("Choose a target pose image", type=["jpg", "jpeg", "png"])
    transform_button = st.button("Transform")
    if target_file is not None:
        # Display the target pose image
        target_image = st.image(target_file, caption="Target Pose Image", use_column_width=True)
        st.session_state.target_filename = target_file.name
        target_image = Image.open(target_file)
        target_image.save(UPLOADED_FILE + "/" + f"{st.session_state.target_filename}")

    if transform_button:
        perform_pose_transfer(st.session_state.filename, st.session_state.target_filename)


def perform_pose_transfer(filename, target_image_path):
    pose_transfer_dir = "\pose_transfer"
    pid = (f"{filename}", None, None)
    pose_id = (f"{target_image_path}", None, None)

    # Call pose_transfer function
    pose_transfer(pid=pid, pose_id=pose_id, dir=pose_transfer_dir)

    # Save the processed image
    st.success(f"Processed image '{filename}' transformed successfully!")
    st.subheader("Output")
    st.image(OUTPUT_FILE + pose_transfer_dir + "/" + f"{filename}")


# LAYERING
def layering_parameters():
    global target_image  # Make variable global

    st.subheader("Layering Parameters")
    check = [0, 0, 0]
    gids_file = []
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Top")
        top_file = st.file_uploader("Choose garments image", type=["jpg", "jpeg", "png"], key="top_file")
    if top_file is not None:
        gids_file.append(top_file)
        check[0] = 1
    else:
        gids_file.append(-1)

    with col2:
        st.header("Bottom")
        bottom_file = st.file_uploader("Choose garments image", type=["jpg", "jpeg", "png"], key="bottom_file")
    if bottom_file is not None:
        gids_file.append(bottom_file)
        check[1] = 1
    else:
        gids_file.append(-1)

    with col3:
        st.header("Jacket")
        jacket_file = st.file_uploader("Choose garments image", type=["jpg", "jpeg", "png"], key="jacket_file")
    if jacket_file is not None:
        gids_file.append(jacket_file)
        check[2] = 1
    else:
        gids_file.append(-1)

    transform_button = st.button("Transform")

    gids = []
    if gids_file is not None:
        # Display the target pose image
        for i, file in enumerate(gids_file):
            if file != -1:
                st.image(file, caption=f"{file.name}", use_column_width=True)
                g_file = Image.open(file)
                g_file.save(UPLOADED_FILE + "/" + f"{file.name}")
                if i == 0 and gids_file[i] != -1:
                    gids.append((f"{file.name}", None, 5))
                else:
                    pass

                if i == 1 and gids_file[i] != -1:
                    gids.append((f"{file.name}", None, 1))
                else:
                    pass

                if i == 2 and gids_file[i] != -1:
                    gids.append((f"{file.name}", None, 3))

    if transform_button:
        perform_layering(st.session_state.filename, gids)


def perform_layering(filename, gids):
    layering_dir = "\layering"
    pid = (f"{filename}", None, None)
    # Call pose_transfer function
    layering(pid=pid, gids=gids, dir=layering_dir)

    # Save the processed image
    st.success(f"Processed image '{filename}' transformed successfully!")
    st.subheader("Output")
    st.image(OUTPUT_FILE + layering_dir + "/" + f"{filename}")


# MULTI-LAYER
def multi_layering_parameters():
    global target_image  # Make variable global

    st.subheader("Multi Layering Parameters")
    check = [0, 0, 0]

    # GIDS LIST
    gids_file = []
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Top")
        top_file = st.file_uploader("Choose garments image", type=["jpg", "jpeg", "png"], key="top_file")
    if top_file is not None:
        gids_file.append(top_file)
        check[0] = 1
    else:
        gids_file.append(-1)

    with col2:
        st.header("Bottom")
        bottom_file = st.file_uploader("Choose garments image", type=["jpg", "jpeg", "png"], key="bottom_file")
    if bottom_file is not None:
        gids_file.append(bottom_file)
        check[1] = 1
    else:
        gids_file.append(-1)

    with col3:
        st.header("Jacket")
        jacket_file = st.file_uploader("Choose garments image", type=["jpg", "jpeg", "png"], key="jacket_file")
    if jacket_file is not None:
        gids_file.append(jacket_file)
        check[2] = 1
    else:
        gids_file.append(-1)

    gids = []
    if gids_file is not None:
        # Display the target pose image
        for i, file in enumerate(gids_file):
            if file != -1:
                st.image(file, caption=f"{file.name}", use_column_width=True)
                g_file = Image.open(file)
                g_file.save(UPLOADED_FILE + "/" + f"{file.name}")
                if i == 0 and gids_file[i] != -1:
                    gids.append((f"{file.name}", None, 5))
                else:
                    pass

                if i == 1 and gids_file[i] != -1:
                    gids.append((f"{file.name}", None, 1))
                else:
                    pass

                if i == 2 and gids_file[i] != -1:
                    gids.append((f"{file.name}", None, 3))

    # OGIDS LIST
    ogids_file = []
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Top")
        top_file = st.file_uploader("Choose garments image", type=["jpg", "jpeg", "png"], key="otop_file")
    if top_file is not None:
        ogids_file.append(top_file)
        check[0] = 1
    else:
        ogids_file.append(-1)

    with col2:
        st.header("Bottom")
        bottom_file = st.file_uploader("Choose garments image", type=["jpg", "jpeg", "png"], key="obottom_file")
    if bottom_file is not None:
        ogids_file.append(bottom_file)
        check[1] = 1
    else:
        ogids_file.append(-1)

    with col3:
        st.header("Jacket")
        jacket_file = st.file_uploader("Choose garments image", type=["jpg", "jpeg", "png"], key="ojacket_file")
    if jacket_file is not None:
        ogids_file.append(jacket_file)
        check[2] = 1
    else:
        ogids_file.append(-1)

    transform_button = st.button("Transform")

    ogids = []
    if ogids_file is not None:
        # Display the target pose image
        for i, file in enumerate(ogids_file):
            if file != -1:
                st.image(file, caption=f"{file.name}", use_column_width=True)
                og_file = Image.open(file)
                og_file.save(UPLOADED_FILE + "/" + f"{file.name}")
                if i == 0 and ogids_file[i] != -1:
                    ogids.append((f"{file.name}", None, 5))
                else:
                    pass

                if i == 1 and ogids_file[i] != -1:
                    ogids.append((f"{file.name}", None, 1))
                else:
                    pass

                if i == 2 and ogids_file[i] != -1:
                    ogids.append((f"{file.name}", None, 3))

    if transform_button:
        perform_multi_layering(st.session_state.filename, gids, ogids)


def perform_multi_layering(filename, gids, ogids):
    multi_layering_dir = "\multi_layering"
    pid = (f"{filename}", None, None)
    # Call pose_transfer function
    layering(pid=pid, gids=gids, ogids=ogids, dir=multi_layering_dir, multi_layer=1)

    # Save the processed image
    st.success(f"Processed image '{filename}' transformed successfully!")
    st.subheader("Output")
    st.image(OUTPUT_FILE + multi_layering_dir + "/" + f"{filename}")

# TUCKING-IN TUCKING-OUT
def tuckinout_parameter():
    global target_image  # Make variable global

    st.subheader("Tucking in/Tucking out Parameters")
    check = [0, 0]
    gids_file = []
    col1, col2 = st.columns(2)
    with col1:
        st.header("Top")
        top_file = st.file_uploader("Choose garments image", type=["jpg", "jpeg", "png"], key="top_file")
    if top_file is not None:
        gids_file.append(top_file)
        check[0] = 1
    else:
        gids_file.append(-1)

    with col2:
        st.header("Bottom")
        bottom_file = st.file_uploader("Choose garments image", type=["jpg", "jpeg", "png"], key="bottom_file")
    if bottom_file is not None:
        gids_file.append(bottom_file)
        check[1] = 1
    else:
        gids_file.append(-1)

    transform_button = st.button("Transform")

    gids = []
    option = st.radio("Select Processing Option", ["Tucking in", "Tucking out"])
    if gids_file is not None:
        # Display the target pose image
        for i, file in enumerate(gids_file):
            if file != -1:
                st.image(file, caption=f"{file.name}", use_column_width=True)
                g_file = Image.open(file)
                g_file.save(UPLOADED_FILE + "/" + f"{file.name}")
                if i == 0 and gids_file[i] != -1:
                    gids.append((f"{file.name}", None, 5))
                else:
                    pass

                if i == 1 and gids_file[i] != -1:
                    gids.append((f"{file.name}", None, 1))
                else:
                    pass

    flag = 0
    if option == "Tucking in":
        flag = 1
    else:
        flag = 0
    if transform_button:
        perform_tuckinout(st.session_state.filename, gids, flag)


def perform_tuckinout(filename, gids, flag):
    layering_dir = "\layer_tucking_in_out"
    pid = (f"{filename}", None, None)
    # Call pose_transfer function
    tucking_in_out(pid=pid, gids=gids, dir=layering_dir, tuking_in=flag)

    # Save the processed image
    st.success(f"Processed image '{filename}' transformed successfully!")
    st.subheader("Output")
    st.image(OUTPUT_FILE + layering_dir + "/" + f"{filename}")

if __name__ == "__main__":
    main()