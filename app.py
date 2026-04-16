import streamlit as st

from answerer import answer_question


st.set_page_config(page_title="Hybrid Math Reading Assistant", layout="wide")

st.title("Hybrid Math Reading Assistant")
st.write("Ask a question about your mathematical document library.")

question = st.text_input(
    "Enter your question / 输入你的问题：",
    placeholder="e.g. What is exchangeability? / exchangeable怎么定义的"
)

answer_style = st.selectbox(
    "Answer style",
    ["Technical", "Plain-language"]
)

include_example = st.checkbox("Include a simple example", value=False)

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        result = answer_question(
            question=question,
            top_k=5,
            answer_style=answer_style,
            include_example=include_example,
        )

    st.subheader("Mode")
    if result["mode"] == "Library-grounded":
        st.success(result["mode"])
    elif result["mode"] == "Hybrid":
        st.warning(result["mode"])
    else:
        st.error(result["mode"])

    st.subheader("Routing Reason")
    st.write(result["reason"])

    st.subheader("Answer")
    st.markdown(result["answer"])

    st.subheader("Retrieved Source Snippets")

    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    distances = result.get("distances", [])

    if documents:
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
            source = meta.get("source", "unknown")
            chunk_index = meta.get("chunk_index", "unknown")

            with st.expander(f"Source {i}: {source} | chunk {chunk_index} | distance={dist:.4f}"):
                st.write(doc[:1200])
    else:
        st.write("No retrieved snippets available.")