using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEngine.UI;
using UnityEngine.SceneManagement;


public class UIManager : MonoBehaviour
{

    public GameObject notice00;
    public GameObject notice01;
    public GameObject notice02;

    public GameObject nextButton;



    // Start is called before the first frame update
    void Start()
    {
        notice00.gameObject.SetActive(true);
        notice01.gameObject.SetActive(false);
        notice02.gameObject.SetActive(false);
    }

    public void OnNextButtonClicked()
    {
        if (notice00.activeSelf == true)
        {
            notice00.SetActive(false);
            notice01.SetActive(true);
            notice02.SetActive(false);
        }
        else if(notice01.activeSelf == true)
        {
            notice00.SetActive(false);
            notice01.SetActive(false);
            notice02.SetActive(true);
        }
        else if(notice02.activeSelf == true)
        {
            SceneManager.LoadScene("MainScene");
        }
    }
}
