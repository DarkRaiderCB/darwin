// App.js

import React, { useEffect, useRef, useState } from "react";
import styles from "./Code.module.css";
import { useZustandStore } from "../../store";
import Terminal, {
  ColorMode,
  TerminalInput,
  TerminalOutput,
} from "react-terminal-ui";

import { ReactComponent as SETTING_SVG } from "../../Assets/SVG/setting.svg";
import { ReactComponent as SETTING_SVG_WHITE } from "../../Assets/SVG/setting-white.svg";
import Setting from "../Setting/Setting";
import OtherBoxes from "../Otherboxes/Manager";
import { getCode } from "../../api/getCode";

const Code = () => {
  const {
    messages,
    setMessages,
    selected,
    setselected,
    selectedProject_id,
    selectedProject,
    files,
    setFiles,
    theme,
    editor_expanded,
    setEditor_expanded,
    plan,
    setplan,
  } = useZustandStore();
  const [code, setcode] = useState(
    `print("Here is your personal software engineer 🙂")`
  );
  const [language, setlanguage] = useState("python");
  const [input_msg, setinput_msg] = useState("");
  const [istyping, setistyping] = useState(false);
  const [selected_file, setselected_file] = useState(null);
  const [selected_file_language, setselected_file_language] = useState(
    selected_file?.language || "python"
  );
  const [issettingopen, setissettingopen] = useState(false);

  useEffect(() => {
    console.log("selected file : ", selected_file);
    console.log("selected project : ", selectedProject);
  }, [selected_file, selectedProject]);

  const inputref = useRef(null);

  useEffect(() => {
    inputref?.current?.focus();
  }, []);

  const msgref = useRef(null);
  const msgref2 = useRef(null);

  useEffect(() => {
    msgref?.current?.scrollIntoView({ behavior: "smooth" });
    msgref2?.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, code, language, input_msg]);

  const [colorMode, setColorMode] = useState(ColorMode.Dark);
  const [lineData, setLineData] = useState([
    <TerminalOutput>Output will appear here!!</TerminalOutput>,
    <TerminalOutput>$ ⎕</TerminalOutput>,
  ]);

  // Function to handle sending a message
  const sendMessage = async (message) => {
    if (!selectedProject) {
      alert("Please select the project first.");
      return;
    }
    setistyping(true);
    let msgs = messages;
    msgs.push({ text: message, sender: "user" });
    setMessages(msgs);
    let ld = [<TerminalOutput>Output will appear here!!</TerminalOutput>];
    let codee = "";

    console.log(selectedProject);

    const formData = new FormData();
    formData.append("project_name", selectedProject);
    formData.append("customer_message", message);

    try {
      const res_text = await getCode(formData);

      if (!res_text) {
        console.log("Error in backend.");
        return;
      }

      console.log("Response : ", res_text);

      const reader = res_text.body.getReader();
      const decoder = new TextDecoder("utf-8");

      while (true && reader && decoder) {
        const { done, value } = await reader.read();
        if (done) {
          console.log("Done");
          setistyping(false);
          break;
        }

        const chunk = decoder.decode(value, { stream: true });

        console.log("Line : ", chunk);

        const data = JSON.parse(chunk);

        console.log("Data : ", data);

        if (data?.summary_text) {
          msgs.push({ text: data?.summary_text, sender: "bot" });
          setMessages(msgs);
        }

        if (data?.message) {
          let pl = plan;
          pl.push(`${data?.message}\n\n`);
          setplan(pl);
        }

        if (data?.web_search) {
          let pl = plan;
          pl.push(`${data?.web_search}\n\n`);
          setplan(pl);
        }

        if (data?.console) {
          ld.push(<TerminalInput>{data?.console}</TerminalInput>);
          ld.push(<TerminalInput>{`\n`}</TerminalInput>);
          setLineData(ld);
        }
      }

      // const res_json = JSON.parse(res_text);

      // setistyping(false);

      // console.log("Res : ", res_json);

      // msgs.push({ text: res_json?.summary, sender: "bot" });
      // setMessages(msgs);
      // console.log("Msg : ", res_json?.summary);
      // // }

      // let c = "";
      // let o = "";

      // if (
      //   res_json &&
      //   res_json?.coder_response &&
      //   res_json?.coder_response.length > 0
      // ) {
      //   for (const item of res_json?.coder_response) {
      //     if (item?.output) {
      //       o = item?.output + "\n";
      //     }
      //     if(item?.message){
      //       setplan(item?.message);
      //     }
      //   }
      // }

      // setcode(c);

      // let ld = [<TerminalOutput>Output will appear here!!</TerminalOutput>];
      // ld.push(<TerminalInput>{o}</TerminalInput>);
      // setLineData(ld);
      // console.log("Output : ", o);
    } catch (err) {
      console.log(err);
      setistyping(false);
      console.log("Please try again.");
      return;
    }
  };

  return (
    <div className={`${styles.container} `}>
      <div
        className={`${styles.sidebar} ${
          theme === "Dark" ? styles.darkmode : null
        }`}
      >
        <button
          style={
            theme === "Ligth"
              ? { border: "none", background: "none", cursor: "pointer" }
              : { border: "none", background: "none", color: "white" }
          }
        >
          {theme === "Dark" ? (
            <SETTING_SVG_WHITE
              className={`${styles.sidebarli}`}
              onClick={() => {
                setissettingopen(!issettingopen);
              }}
            />
          ) : (
            <SETTING_SVG
              className={`${styles.sidebarli} `}
              onClick={() => {
                setissettingopen(!issettingopen);
              }}
            />
          )}
        </button>
      </div>
      <div
        className={`${
          editor_expanded ? styles.chatbotBox : styles.chatbotBox2
        }  ${theme === "Dark" && !issettingopen ? styles.darkmode : null}`}
      >
        {/* <div className={styles.chatHeader}>Chatbot</div> */}

        {!issettingopen ? (
          <>
            <div className={`${styles.messageContainer} `}>
              {messages &&
                messages.length > 0 &&
                messages.map((message, index) => {
                  const texts = message?.text?.split("\n");
                  return (
                    <div key={index + 11}>
                      <div
                        key={index + 2}
                        className={`${styles.message} ${
                          styles[message.sender]
                        } ${
                          theme === "Dark"
                            ? message.sender === "user"
                              ? styles.userdarkmode
                              : styles.botdarkmode
                            : null
                        }`}
                      >
                        {texts &&
                          texts.length > 0 &&
                          texts.map((item, index) => {
                            if (
                              item &&
                              index === texts.length - 2 &&
                              !texts[index + 1]
                            ) {
                              return (
                                <>
                                  {item.startsWith("**") &&
                                  item.endsWith("**") ? (
                                    <span key={index}>
                                      <b className={styles.b}>
                                        <span>
                                          {item.substring(2, item.length - 2)}
                                        </span>
                                      </b>
                                    </span>
                                  ) : item.startsWith("#") ? (
                                    <span key={index}>
                                      <b className={styles.b}>
                                        <span>{item.replaceAll("#", "")}</span>
                                      </b>
                                    </span>
                                  ) : (
                                    <span key={index}>
                                      <span>{item}</span>
                                    </span>
                                  )}
                                </>
                              );
                            } else if (item) {
                              if (index !== texts.length - 1) {
                                return (
                                  <>
                                    {item.startsWith("**") &&
                                    item.endsWith("**") ? (
                                      <span key={index}>
                                        <b>
                                          <span>
                                            {item.substring(2, item.length - 2)}
                                          </span>
                                        </b>
                                        <br />
                                        <br />
                                      </span>
                                    ) : item.startsWith("#") ? (
                                      <span key={index}>
                                        <b>
                                          <span>
                                            {item.replaceAll("#", "")}
                                          </span>
                                        </b>
                                        <br />
                                        <br />
                                      </span>
                                    ) : (
                                      <span key={index}>
                                        <span>{item}</span>
                                        <br />
                                        <br />
                                      </span>
                                    )}
                                  </>
                                );
                              } else {
                                return (
                                  <>
                                    {item.startsWith("**") &&
                                    item.endsWith("**") ? (
                                      <span key={index}>
                                        <b className={styles.b}>
                                          <span>
                                            {item.substring(2, item.length - 2)}
                                          </span>
                                        </b>
                                      </span>
                                    ) : item.startsWith("#") ? (
                                      <span key={index}>
                                        <b className={styles.b}>
                                          <span>
                                            {item.replaceAll("#", "")}
                                          </span>
                                        </b>
                                      </span>
                                    ) : (
                                      <span key={index}>
                                        <span>{item}</span>
                                      </span>
                                    )}
                                  </>
                                );
                              }
                            }
                          })}
                      </div>
                    </div>
                  );
                })}
              {!issettingopen && istyping ? (
                <div
                  ref={msgref2}
                  className={`${styles.message} ${styles["bot"]} ${
                    theme === "Dark" ? styles.botdarkmode : null
                  }`}
                >
                  Typing...
                </div>
              ) : (
                ""
              )}
              <div ref={msgref}></div>
              {/* ) : (
            <Setting />
          )} */}
            </div>
            <textarea
              type="text"
              ref={inputref}
              value={input_msg}
              rows={
                input_msg.split(" ").length > 20 || input_msg.length > 50
                  ? 10
                  : 1
              }
              placeholder="Ask Darwin..."
              onChange={(e) => {
                setinput_msg(e.target.value);
              }}
              onKeyPress={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  e.target.value = "";
                  setinput_msg("");
                  sendMessage(input_msg);
                }
              }}
              className={`${styles.input} ${
                theme === "Dark" ? styles.darkmode_input : null
              }`}
            />
          </>
        ) : (
          <Setting />
        )}
      </div>

      <OtherBoxes colorMode={colorMode} lineData={lineData} plan={plan} />
    </div>
  );
};

export default Code;
