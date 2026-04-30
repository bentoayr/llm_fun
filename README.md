### Question: Is it easy to make a machine that improves machines, and use it to improve itself in an ever improving cycle?

#### As a simplistic model of this problem, we will define

    machine/code = prompt that acepts as input text
    physical reality where machine lives/hardware where code runs = fixed LLM
    running a machine, or executing code =  calling the LLM on the prompt with its input text
    improving a machine = improving a prompt

  #### Why do we use this model ?

     * evolving actual machine/code is expensive and complicated

        * the machine/code might not run/compile

        * it might not accept the correct types of inputs or might produce the wrong type of outputs

        * checking if the machine/code has a "useful behavior", even if we taking running/compilation for granted and take producing correct input/ouputs for granted, might not be easy accross all problems

    * on the other hand, evolving text is easy

        * code is text and always executes

        * input format = text, output format = text

        * code is in english and easily (subjectively) be judge 


### You can run some scripts live:

##### https://www.wolframcloud.com/obj/josebento/Published/some_fun_with_recursion_and_llms.nb

##### https://colab.research.google.com/drive/10Tex5cKlNOe5o3qQzbF79FkyGUzhMHpO?usp=sharing

