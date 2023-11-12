import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
#This program extracts the desired variables from the output of training the model to avoid doing things by hand.
#Paste the output into a file starting at the first iteration, and supply the filename in main
#Then the extract method will fill lists of the values of each desired training statistic
#Then you just need to modify the plotting code to customize the plots

def extract(filename,approx_kl,entropy_loss,explained_variance,loss,policy_gradient_loss,value_loss,iterations):
    f = open(filename,"r")
    lines = f.readlines()
    for i in range(39):
        if i > 18 and iterations == 19:
            approx_kl.append(0)
            entropy_loss.append(0)
            explained_variance.append(0)
            loss.append(0)
            policy_gradient_loss.append(0)
            value_loss.append(0)
        else:
            #Add values to list
            addition = (18* i) + 7 #New line calculation between iterations
            print("addition" + str(addition))
            print("Iters" + str(iterations))
            print("i = " + str(i))
            print(lines[7 + addition])
            approx_kl_curr = float(re.findall('\d+\.\d+', lines[7 + addition])[0])
            nums = re.findall('0\d+', lines[7+addition])
            multiplier = int(str(nums[0])[1])
            approx_kl_curr = approx_kl_curr * 10**-multiplier
            approx_kl.append(approx_kl_curr)
            entropy_loss.append(float(re.findall('-?\d+(?:\.\d+)?', lines[10 + addition])[0]))
            explained_variance.append(float(re.findall('-?\d+(?:\.\d+)?', lines[11+ addition])[0]))
            loss.append(float(re.findall('-?\d+(?:\.\d+)?', lines[13+ addition])[0]))
            policy_gradient_loss.append(float(re.findall('-?\d+(?:\.\d+)?', lines[15+ addition])[0]))
            value_loss.append(float(re.findall('-?\d+(?:\.\d+)?', lines[16+ addition])[0]))
        
        

def main():
    iterations = [i for i in range(1,40)]
    #Supply filenames
    filename,approx_kl,entropy_loss,explained_variance,loss,policy_gradient_loss,value_loss = 'run_logs/rundata5.txt',[],[],[],[],[],[]
    filename2,approx_kl2,entropy_loss2,explained_variance2,loss2,policy_gradient_loss2,value_loss2 = 'run_logs/rundata6.txt',[],[],[],[],[],[]
    #Extract data
    extract(filename,approx_kl,entropy_loss,explained_variance,loss,policy_gradient_loss,value_loss,19)
    extract(filename2,approx_kl2,entropy_loss2,explained_variance2,loss2,policy_gradient_loss2,value_loss2,39)

    #Plot data
    #This could be much shorter and better code if it was in a function, but alas there are only so many hours in the day.
    #approx_kl
    fig, ax = plt.subplots() 
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Approx_kl")
    ax.set_title("Approx_kl")
    ax.plot(iterations, approx_kl, marker="o", label="10k Timesteps", drawstyle="steps-post")
    ax.plot(iterations, approx_kl2, marker="o",label="20k Timesteps", drawstyle="steps-post")
    
    ax.legend()
    plt.show()

    #entropy_loss
    fig, ax = plt.subplots() 
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy loss")
    ax.set_title("Entropy loss")
    ax.plot(iterations, entropy_loss, marker="o", label="10k Timesteps", drawstyle="steps-post")
    ax.plot(iterations, entropy_loss2, marker="o",label="20k Timesteps", drawstyle="steps-post")
   
    ax.legend()
    plt.show()

    #explained_variance
    fig, ax = plt.subplots() 
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Explained Variance")
    ax.set_title("Explained Variance")
    ax.plot(iterations, explained_variance, marker="o", label="10k Timesteps" , drawstyle="steps-post")
    ax.plot(iterations, explained_variance2, marker="o",label="20k Timesteps" , drawstyle="steps-post")
    
    ax.legend()
    plt.show()

    #loss
    fig, ax = plt.subplots() 
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.plot(iterations, loss, marker="o", label="10k Timesteps" , drawstyle="steps-post")
    ax.plot(iterations, loss2, marker="o",label="20k Timesteps" , drawstyle="steps-post")
    
    ax.legend()
    plt.show()

    #policy_gradient_loss
    fig, ax = plt.subplots() 
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Policy gradient loss")
    ax.set_title("Policy gradient loss")
    ax.plot(iterations, policy_gradient_loss, marker="o", label="10k Timesteps" , drawstyle="steps-post")
    ax.plot(iterations, policy_gradient_loss2, marker="o",label="20k Timesteps" , drawstyle="steps-post")
    
    ax.legend()
    plt.show()

    #value_loss
    fig, ax = plt.subplots() 
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value Loss")
    ax.set_title("Value Loss")
    ax.plot(iterations, value_loss, marker="o", label="10k Timesteps" , drawstyle="steps-post")
    ax.plot(iterations, value_loss2, marker="o",label="20k Timesteps" , drawstyle="steps-post")
    
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
