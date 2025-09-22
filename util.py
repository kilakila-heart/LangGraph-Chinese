import os.path

from langchain_core.runnables.graph import MermaidDrawMethod

display_graph_path = "E:\\code\\LangGraph\\LangGraph-Chinese\\display_graph"

'''
因为ipython画图的时候要调用外部接口，有时候外部接口挂掉了，因此本地方式更合适
'''
def get_langgraph_display(graph, current_file_name):
    # 本地 / fallback 渲染
    current_file_name = current_file_name + ".jpg"
    try:
        img_bytes = graph.get_graph().draw_mermaid_png(
            # 指定方法为 Pyppeteer，本地渲染
            draw_method=MermaidDrawMethod.PYPPETEER,
            max_retries=5,
            retry_delay=2.0,
            background_color="white",
            padding=10
        )
        # 保存文件或显示
        with open(os.path.join(display_graph_path, current_file_name), "wb") as f:
            f.write(img_bytes)
        print("图表已保存到 {0}".format(current_file_name))
    except Exception as e:
        print("渲染图表失败:", e)
        # fallback：打印 mermaid 源代码
        mermaid_src = graph.get_graph().draw_mermaid()
        print("Mermaid 源码如下：")
        print(mermaid_src)
